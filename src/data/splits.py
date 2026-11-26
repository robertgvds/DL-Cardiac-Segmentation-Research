"""Gerador de splits estratificados por paciente para experimentos nnU-Net v2.

Garante ausência total de vazamento de dados (zero data leakage) mantendo o
split estratificado oficial `SEED=42` usado nos baselines (Exp 02B UKBB e
Exp 03 U-Net ResNet34) para comparação científica estritamente justa.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    """Constrói `splits_final.json` estratificado no nível de paciente para `nnUNetv2`."""

    def __init__(self, config: BenchmarkSplitConfig | None = None) -> None:
        self.config = config or BenchmarkSplitConfig()

    def build_nnunet_splits(
        self,
        train_case_scar_map: Dict[str, bool],
        dry_run: bool = False,
    ) -> List[Dict[str, List[str]]]:
        """Gera lista de folds no formato oficial `splits_final.json` da nnU-Net v2.

        Quando todos os 5 casos de validação de referência (`SEED=42`) estão
        presentes em `train_case_scar_map`, o **Fold 0** replica exatamente a
        divisão 35 treino / 5 validação dos baselines anteriores, e os Folds
        1..4 completam a validação cruzada estratificada sobre os 40 casos.

        Args:
            train_case_scar_map: Dicionário `{case_id: has_scar}` apenas dos
                casos contidos em `imagesTr` (excluindo `imagesTs`).
            dry_run: Se True, permite conjuntos mínimos (ex.: 2 volumes em
                `data_sample/`) gerando split sintético válido para teste em CPU.

        Returns:
            Lista de dicionários `{"train": [...], "val": [...]}` compatível com
            `nnUNet_preprocessed/DatasetXXX/splits_final.json`.
        """
        all_ids = sorted(train_case_scar_map.keys())
        if len(all_ids) < 2:
            raise ValueError(f"São necessários pelo menos 2 casos em imagesTr; encontrados {len(all_ids)}.")

        # Verifica se há vazamento com o conjunto de teste fixo
        leaked = set(all_ids).intersection(self.config.test_cases)
        if leaked:
            raise ValueError(f"Data leakage detectado! Casos de teste presentes em imagesTr: {sorted(leaked)}")

        if dry_run or len(all_ids) < self.config.num_folds:
            # Modo de validação rápida em CPU (data_sample com 2 volumes)
            val_id = [all_ids[-1]]
            train_ids = all_ids[:-1]
            LOGGER.info("Split dry-run gerado: train=%s | val=%s", train_ids, val_id)
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

        # Distribui os casos restantes de forma estratificada por presença de fibrose
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

        LOGGER.info(
            "Gerados %d folds estratificados (Fold 0 espelha SEED=42: %s | train=%d, val=%d).",
            len(splits),
            use_exact_fold0,
            len(splits[0]["train"]),
            len(splits[0]["val"]),
        )
        return splits

    def save_splits_json(self, splits: Sequence[Dict[str, List[str]]], output_path: Path) -> Path:
        """Persiste `splits_final.json` formatado no diretório alvo."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(list(splits), indent=2), encoding="utf-8")
        LOGGER.info("Arquivo splits_final.json salvo em %s", output_path)
        return output_path
