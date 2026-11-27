"""Runner fino para o Exp 06: Cascata em 2 Etapas (Anatomia Exp 05A -> Especialista Fibrose CMR+EMIDEC).

Monta os 3 datasets privados (`cmr-multi-lge-sax`, `emidec-lge-sax`, `hu-ufjf-lge-sax`)
e reaproveita diretamente o `checkpoint_best.pth` do kernel `robertgvds/dlcs-unet-resnet34-4c`
como Etapa 1 (economizando GPU e garantindo comparação estritamente controlada).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

LOGGER = logging.getLogger("dlcs_cascade_scar_emidec_runner")


@dataclass(frozen=True)
class KaggleCascadeJobConfig:
    """Parâmetros imutáveis do job `dlcs-cascade-scar-emidec`."""

    repo_url: str = "https://github.com/robertgvds/DL-Cardiac-Segmentation-Research.git"
    branch: str = "exp/cascade-scar-cmr-emidec"
    cmr_slug: str = "cmr-multi-lge-sax"
    emidec_slug: str = "emidec-lge-sax"
    hu_slug: str = "hu-ufjf-lge-sax"
    stage1_kernel_slug: str = "dlcs-unet-resnet34-4c"
    kaggle_owner: str = "robertgvds"
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 3e-4
    seed: int = 42
    kaggle_input_root: Path = Path("/kaggle/input")
    scratch_dir: Path = Path("/tmp/dlcs_cascade_emidec")
    working_dir: Path = Path("/kaggle/working")


class KaggleCascadeScarRunner:
    """Orquestrador fino do Exp 06 na GPU do Kaggle."""

    def __init__(self, config: KaggleCascadeJobConfig | None = None) -> None:
        self.config = config or KaggleCascadeJobConfig()
        self.repo_dir = self.config.scratch_dir / "repo"
        self.export_dir = self.config.working_dir / "export"

    def _resolve_input_dir(self, slug: str) -> Path:
        """Resolve o diretório montado no Kaggle (padrão novo datasets/notebooks e legado)."""
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

    def _resolve_stage1_checkpoint(self) -> Optional[Path]:
        """Localiza `checkpoint_best.pth` montado a partir do kernel Exp 05A (`dlcs-unet-resnet34-4c`)."""
        try:
            kdir = self._resolve_input_dir(self.config.stage1_kernel_slug)
        except FileNotFoundError:
            LOGGER.warning("Kernel da Etapa 1 não montado; fallback para busca global em /kaggle/input.")
            kdir = self.config.kaggle_input_root

        for ckpt in sorted(kdir.rglob("checkpoint_best.pth")):
            if ckpt.is_file():
                LOGGER.info("Checkpoint da Etapa 1 localizado em: %s", ckpt)
                return ckpt
        LOGGER.warning("checkpoint_best.pth não encontrado; Etapa 1 será treinada do zero.")
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
        """Executa o pipeline da Cascata em 2 Etapas (CMR-MULTI + EMIDEC -> HU-UFJF)."""
        self._setup_environment()
        cmr_dir = self._resolve_input_dir(self.config.cmr_slug)
        emidec_dir = self._resolve_input_dir(self.config.emidec_slug)
        hu_dir = self._resolve_input_dir(self.config.hu_slug)
        stage1_ckpt = self._resolve_stage1_checkpoint()

        cmd = [
            sys.executable,
            "-m",
            "src.models.cascade_scar",
            "--cmr-dir",
            str(cmr_dir),
            "--emidec-dir",
            str(emidec_dir),
            "--hu-dir",
            str(hu_dir),
            "--output-dir",
            str(self.export_dir),
            "--epochs",
            str(self.config.epochs),
            "--batch-size",
            str(self.config.batch_size),
            "--lr",
            str(self.config.learning_rate),
            "--seed",
            str(self.config.seed),
        ]
        if stage1_ckpt is not None:
            cmd.extend(["--stage1-checkpoint", str(stage1_ckpt)])

        self._run_cmd(cmd, cwd=self.repo_dir)
        LOGGER.info("Job dlcs-cascade-scar-emidec concluído. Artefatos em %s", self.export_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    KaggleCascadeScarRunner().run()
