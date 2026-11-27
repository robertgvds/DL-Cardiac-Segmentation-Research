"""Runner fino para o Exp 05B: U-Net ResNet34 (5 classes — Anatomia + Scar) no Kaggle.

Treina no CMR-MULTI LGE SAX com todas as 5 classes (`0=Bg, 1=VE, 2=MioSaudavel,
3=Scar, 4=VD`), avalia no conjunto de teste inédito do CMR-MULTI (N=7, `SEED=42`)
e na coorte clínica privada HU-UFJF (N=10 rotulados padronizados + `patient_20`),
exportando NIfTIs (`int16`, affine preservada), métricas 3D (clínicas + MICCAI 2026)
e figuras comparativas.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger("dlcs_unet_resnet34_5c_runner")


@dataclass(frozen=True)
class KaggleUnetJobConfig:
    """Parâmetros imutáveis do job `dlcs-unet-resnet34-5c`."""

    repo_url: str = "https://github.com/robertgvds/DL-Cardiac-Segmentation-Research.git"
    branch: str = "exp/unet-resnet34-cmr-multi"
    task_mode: str = "full_5c"
    cmr_slug: str = "cmr-multi-lge-sax"
    hu_slug: str = "hu-ufjf-lge-sax"
    kaggle_owner: str = "robertgvds"
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 3e-4
    patch_size: int = 256
    target_spacing_mm: float = 1.0
    seed: int = 42
    kaggle_input_root: Path = Path("/kaggle/input")
    scratch_dir: Path = Path("/tmp/dlcs_unet_5c")
    working_dir: Path = Path("/kaggle/working")


class KaggleUnetResNet34Runner:
    """Orquestrador fino para execução na GPU do Kaggle mantendo `/kaggle/working/export/` limpo."""

    def __init__(self, config: KaggleUnetJobConfig | None = None) -> None:
        self.config = config or KaggleUnetJobConfig()
        self.repo_dir = self.config.scratch_dir / "repo"
        self.export_dir = self.config.working_dir / "export"

    def _resolve_input_dataset(self, slug: str) -> Path:
        """Resolve o diretório montado do dataset no Kaggle (padrão novo e legado)."""
        candidates = [
            self.config.kaggle_input_root / "datasets" / self.config.kaggle_owner / slug,
            self.config.kaggle_input_root / slug,
        ]
        for cand in candidates:
            if cand.exists():
                LOGGER.info("Dataset '%s' localizado em: %s", slug, cand)
                return cand

        if self.config.kaggle_input_root.exists():
            for path in sorted(self.config.kaggle_input_root.rglob(slug)):
                if path.is_dir():
                    LOGGER.info("Dataset '%s' localizado via busca dinâmica em: %s", slug, path)
                    return path

        raise FileNotFoundError(f"Dataset '{slug}' não encontrado em {self.config.kaggle_input_root}")

    @staticmethod
    def _run_cmd(cmd: Sequence[str], cwd: Path | None = None) -> None:
        """Executa comando de shell validando código de retorno."""
        LOGGER.info(">>> %s", " ".join(cmd))
        subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)

    def _setup_environment(self) -> None:
        """Clona a branch do experimento em `/tmp` e instala dependências mínimas."""
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
        self._run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "segmentation-models-pytorch",
                "nibabel",
            ]
        )
        subprocess.run(["nvidia-smi"], check=False)

    def run(self) -> None:
        """Dispara o pipeline POO versionado em `src/models/unet_resnet34.py`."""
        self._setup_environment()
        cmr_dir = self._resolve_input_dataset(self.config.cmr_slug)
        hu_dir = self._resolve_input_dataset(self.config.hu_slug)

        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.models.unet_resnet34",
                "--cmr-dir",
                str(cmr_dir),
                "--hu-dir",
                str(hu_dir),
                "--output-dir",
                str(self.export_dir),
                "--task-mode",
                self.config.task_mode,
                "--epochs",
                str(self.config.epochs),
                "--batch-size",
                str(self.config.batch_size),
                "--lr",
                str(self.config.learning_rate),
                "--patch-size",
                str(self.config.patch_size),
                "--target-spacing-mm",
                str(self.config.target_spacing_mm),
                "--seed",
                str(self.config.seed),
            ],
            cwd=self.repo_dir,
        )
        LOGGER.info("Job dlcs-unet-resnet34-5c concluído. Artefatos em %s", self.export_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    KaggleUnetResNet34Runner().run()
