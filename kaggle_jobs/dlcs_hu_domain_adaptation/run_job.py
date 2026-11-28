"""Runner fino para o Exp 07: Adaptação de Domínio HU-UFJF em CV 5-Fold (FT Puro vs. Co-FT).

Reaproveita diretamente o `checkpoint_best.pth` do Exp 05B (`robertgvds/dlcs-unet-resnet34-5c`)
como inicialização *warm-start* e executa CV 5-fold estratificado na coorte clínica HU-UFJF,
comparando Fine-Tuning Puro (`Exp 07A`) vs. Co-Fine-Tuning HU+CMR-MULTI (`Exp 07B`).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

LOGGER = logging.getLogger("dlcs_hu_domain_adaptation_runner")


@dataclass(frozen=True)
class KaggleHuAdaptationJobConfig:
    """Parâmetros imutáveis do job `dlcs-hu-domain-adaptation`."""

    repo_url: str = "https://github.com/robertgvds/DL-Cardiac-Segmentation-Research.git"
    branch: str = "exp/hu-ufjf-domain-adaptation"
    cmr_slug: str = "cmr-multi-lge-sax"
    hu_slug: str = "hu-ufjf-lge-sax"
    base_kernel_slug: str = "dlcs-unet-resnet34-5c"
    kaggle_owner: str = "robertgvds"
    finetune_epochs: int = 60
    co_finetune_epochs: int = 45
    batch_size: int = 16
    seed: int = 42
    kaggle_input_root: Path = Path("/kaggle/input")
    scratch_dir: Path = Path("/tmp/dlcs_hu_adapt")
    working_dir: Path = Path("/kaggle/working")


class KaggleHuAdaptationRunner:
    """Orquestrador fino do Exp 07 na GPU do Kaggle."""

    def __init__(self, config: KaggleHuAdaptationJobConfig | None = None) -> None:
        self.config = config or KaggleHuAdaptationJobConfig()
        self.repo_dir = self.config.scratch_dir / "repo"
        self.export_dir = self.config.working_dir / "export"

    def _resolve_input_dir(self, slug: str) -> Path:
        candidates = [
            self.config.kaggle_input_root / "datasets" / self.config.kaggle_owner / slug,
            self.config.kaggle_input_root / "notebooks" / self.config.kaggle_owner / slug,
            self.config.kaggle_input_root / slug,
        ]
        for cand in candidates:
            if cand.exists():
                LOGGER.info("Input '%s' localizado em: %s", slug, cand)
                return cand
        if self.config.kaggle_input_root.exists():
            for path in sorted(self.config.kaggle_input_root.rglob(slug)):
                if path.is_dir():
                    LOGGER.info("Input '%s' localizado via busca dinâmica em: %s", slug, path)
                    return path
        raise FileNotFoundError(f"Input '{slug}' não encontrado em {self.config.kaggle_input_root}")

    def _resolve_base_checkpoint(self) -> Optional[Path]:
        try:
            kdir = self._resolve_input_dir(self.config.base_kernel_slug)
        except FileNotFoundError:
            kdir = self.config.kaggle_input_root
        for ckpt in sorted(kdir.rglob("checkpoint_best.pth")):
            if ckpt.is_file():
                LOGGER.info("Checkpoint Exp 05B localizado em: %s", ckpt)
                return ckpt
        LOGGER.warning("Checkpoint Exp 05B não encontrado; iniciando de ImageNet.")
        return None

    @staticmethod
    def _run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
        LOGGER.info(">>> %s", " ".join(cmd))
        subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)

    def _setup_environment(self) -> None:
        if self.repo_dir.exists():
            shutil.rmtree(self.repo_dir)
        self.config.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self._run_cmd(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                self.config.branch,
                self.config.repo_url,
                str(self.repo_dir),
            ]
        )
        self._run_cmd([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch", "nibabel"])
        subprocess.run(["nvidia-smi"], check=False)

    def run(self) -> None:
        self._setup_environment()
        cmr_dir = self._resolve_input_dir(self.config.cmr_slug)
        hu_dir = self._resolve_input_dir(self.config.hu_slug)
        base_ckpt = self._resolve_base_checkpoint()

        cmd = [
            sys.executable,
            "-m",
            "src.models.hu_domain_adaptation",
            "--cmr-dir",
            str(cmr_dir),
            "--hu-dir",
            str(hu_dir),
            "--output-dir",
            str(self.export_dir),
            "--finetune-epochs",
            str(self.config.finetune_epochs),
            "--co-finetune-epochs",
            str(self.config.co_finetune_epochs),
            "--batch-size",
            str(self.config.batch_size),
            "--seed",
            str(self.config.seed),
        ]
        if base_ckpt is not None:
            cmd.extend(["--pretrained-checkpoint", str(base_ckpt)])

        self._run_cmd(cmd, cwd=self.repo_dir)
        LOGGER.info("Job dlcs-hu-domain-adaptation concluído. Artefatos em %s", self.export_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    KaggleHuAdaptationRunner().run()
