"""Gerador de splits estratificados por paciente para experimentos em LGE-CMR.

Garante ausência total de vazamento de dados (zero data leakage) mantendo o
split estratificado oficial `SEED=42` usado nos baselines (Exp 02B UKBB,
Exp 03 U-Net ResNet34 e Exp 04 nnU-Net v2) para comparação científica justa.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSplitConfig:
    """Constantes imutáveis do split de referência CMR-MULTI (`SEED=42`)."""

    seed: int = 42
    num_folds: int = 5
    test_cases: Tuple[str, ...] = (
        "LGE_SAX_005",
        "LGE_SAX_007",
        "LGE_SAX_008",
        "LGE_SAX_014",
        "LGE_SAX_026",
        "LGE_SAX_028",
        "LGE_SAX_047",
    )
    val_cases_fold0: Tuple[str, ...] = (
        "LGE_SAX_003",
        "LGE_SAX_010",
        "LGE_SAX_011",
        "LGE_SAX_032",
        "LGE_SAX_034",
    )


class StratifiedSplitGenerator:
    """Constrói splits estratificados no nível de paciente (PyTorch e nnU-Net v2)."""

    def __init__(self, config: BenchmarkSplitConfig | None = None) -> None:
        self.config = config or BenchmarkSplitConfig()

    def split_train_val_test(
        self,
        all_case_scar_map: Dict[str, bool],
        dry_run: bool = False,
    ) -> Dict[str, List[str]]:
        """Divide todos os casos rotulados em `train`, `val` e `test` sem data leakage.

        Replica integralmente o split oficial `35 / 5 / 7` (`SEED=42`) quando
        os casos de referência estão presentes. No modo `dry_run` (ex.: 2 volumes
        em `data_sample/`), cria partição mínima determinística para validação em CPU.

        Args:
            all_case_scar_map: Dicionário `{case_id: has_scar}` dos volumes disponíveis.
            dry_run: Se True, permite subconjuntos mínimos para validação sintética.

        Returns:
            Dicionário com chaves `"train"`, `"val"` e `"test"`.
        """
        all_ids = sorted(all_case_scar_map.keys())
        if len(all_ids) < 2:
            raise ValueError(f"São necessários pelo menos 2 casos; encontrados {len(all_ids)}.")

        if dry_run or len(all_ids) < 12:
            # Em data_sample (2 volumes), usa o primeiro para treino e o último para val/test
            train_ids = all_ids[:-1]
            eval_ids = [all_ids[-1]]
            LOGGER.info(
                "Split dry-run gerado: train=%s | val=%s | test=%s",
                train_ids,
                eval_ids,
                eval_ids,
            )
            return {"train": train_ids, "val": eval_ids, "test": eval_ids}

        id_set = set(all_ids)
        test_set = set(self.config.test_cases)
        val_set = set(self.config.val_cases_fold0)

        if not test_set.issubset(id_set) or not val_set.issubset(id_set):
            missing = sorted((test_set | val_set) - id_set)
            raise ValueError(f"Casos oficiais de val/test ausentes no dataset: {missing}")

        train_ids = sorted(cid for cid in all_ids if cid not in test_set and cid not in val_set)
        val_ids = sorted(self.config.val_cases_fold0)
        test_ids = sorted(self.config.test_cases)

        # Garante disjunção estrita entre partições no nível de paciente
        assert set(train_ids).isdisjoint(val_set)
        assert set(train_ids).isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

        LOGGER.info(
            "Split oficial SEED=%d aplicado: train=%d (%d scar) | val=%d (%d scar) | test=%d (%d scar)",
            self.config.seed,
            len(train_ids),
            sum(all_case_scar_map[c] for c in train_ids),
            len(val_ids),
            sum(all_case_scar_map[c] for c in val_ids),
            len(test_ids),
            sum(all_case_scar_map[c] for c in test_ids),
        )
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    def build_nnunet_splits(
        self,
        train_case_scar_map: Dict[str, bool],
        dry_run: bool = False,
    ) -> List[Dict[str, List[str]]]:
        """Gera lista de folds no formato oficial `splits_final.json` da nnU-Net v2."""
        all_ids = sorted(train_case_scar_map.keys())
        if len(all_ids) < 2:
            raise ValueError(f"São necessários pelo menos 2 casos em imagesTr; encontrados {len(all_ids)}.")

        leaked = set(all_ids).intersection(self.config.test_cases)
        if leaked:
            raise ValueError(f"Data leakage detectado! Casos de teste presentes em imagesTr: {sorted(leaked)}")

        if dry_run or len(all_ids) < self.config.num_folds:
            val_id = [all_ids[-1]]
            train_ids = all_ids[:-1]
            LOGGER.info("Split dry-run nnU-Net gerado: train=%s | val=%s", train_ids, val_id)
            return [{"train": train_ids, "val": val_id}]

        val0_set = set(self.config.val_cases_fold0)
        use_exact_fold0 = val0_set.issubset(set(all_ids))

        splits: List[Dict[str, List[str]]] = []
        if use_exact_fold0:
            fold0_val = sorted(self.config.val_cases_fold0)
            fold0_train = sorted(cid for cid in all_ids if cid not in val0_set)
            splits.append({"train": fold0_train, "val": fold0_val})
            remaining_ids = fold0_train
            extra_folds = self.config.num_folds - 1
        else:
            remaining_ids = all_ids
            extra_folds = self.config.num_folds

        scar_pos = [cid for cid in remaining_ids if train_case_scar_map[cid]]
        scar_neg = [cid for cid in remaining_ids if not train_case_scar_map[cid]]

        rng = random.Random(self.config.seed)
        rng.shuffle(scar_pos)
        rng.shuffle(scar_neg)

        buckets: List[List[str]] = [[] for _ in range(extra_folds)]
        for idx, cid in enumerate(scar_pos):
            buckets[idx % extra_folds].append(cid)
        for idx, cid in enumerate(scar_neg):
            buckets[idx % extra_folds].append(cid)

        for bucket in buckets:
            val_ids = sorted(bucket)
            val_lookup = set(val_ids)
            train_ids = sorted(cid for cid in all_ids if cid not in val_lookup)
            splits.append({"train": train_ids, "val": val_ids})

        return splits

    @staticmethod
    def save_splits_json(splits: object, output_path: Path) -> Path:
        """Salva splits em JSON formatado para rastreabilidade completa."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
        LOGGER.info("Arquivo de splits salvo em %s", output_path)
        return output_path
