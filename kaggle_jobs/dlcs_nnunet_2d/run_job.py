"""Runner fino para treino e avaliação da nnU-Net v2 2D na GPU do Kaggle.

Toda a lógica de preparação de dados, splits estratificados, métricas 3D e
visualização permanece versionada no GitHub (`DL-Cardiac-Segmentation-Research`).
Este script apenas orquestra o ambiente Kaggle, executa a nnU-Net v2 e exporta
artefatos leves + ZIP de predições NIfTI para inspeção no 3D Slicer.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

LOGGER = logging.getLogger("dlcs_nnunet_2d_runner")


@dataclass(frozen=True)
class KaggleJobConfig:
    """Parâmetros imutáveis do job `dlcs-nnunet-2d`."""

    repo_url: str = "https://github.com/robertgvds/DL-Cardiac-Segmentation-Research.git"
    branch: str = "exp/nnunet-2d-cmr-multi"
    dataset_id: int = 501
    dataset_name: str = "Dataset501_CMRMULTI_LGE"
    dataset_slug: str = "cmr-multi-lge-sax"
    kaggle_owner: str = "robertgvds"
    configuration: str = "2d"
    fold: int = 0
    trainer: str = "nnUNetTrainer_250epochs"
    kaggle_input_root: Path = Path("/kaggle/input")
    scratch_dir: Path = Path("/tmp/dlcs_work")
    working_dir: Path = Path("/kaggle/working")


class KaggleNnUNet2DRunner:
    """Orquestrador fino do experimento Exp 04 (`nnUNetv2 2D`) no Kaggle."""

    def __init__(self, config: KaggleJobConfig | None = None) -> None:
        self.config = config or KaggleJobConfig()
        # Mantém repo, raw e preprocessed em /tmp para não poluir os outputs de /kaggle/working
        self.repo_dir = self.config.scratch_dir / "repo"
        self.nnunet_raw = self.config.scratch_dir / "nnUNet_raw"
        self.nnunet_preprocessed = self.config.scratch_dir / "nnUNet_preprocessed"
        self.nnunet_results = self.config.scratch_dir / "nnUNet_results"
        self.preds_dir = self.config.scratch_dir / "preds_ts"
        self.export_dir = self.config.working_dir / "export"

    def _resolve_input_dataset_dir(self) -> Path:
        """Resolve o diretório montado do dataset no Kaggle (padrão novo e legado)."""
        candidates = [
            self.config.kaggle_input_root / "datasets" / self.config.kaggle_owner / self.config.dataset_slug,
            self.config.kaggle_input_root / self.config.dataset_slug,
        ]
        for cand in candidates:
            if cand.exists():
                LOGGER.info("Dataset localizado em: %s", cand)
                return cand

        # Busca dinâmica de fallback dentro de /kaggle/input caso a hierarquia mude novamente
        if self.config.kaggle_input_root.exists():
            for path in sorted(self.config.kaggle_input_root.rglob(self.config.dataset_slug)):
                if path.is_dir():
                    LOGGER.info("Dataset localizado via busca dinâmica em: %s", path)
                    return path

        raise FileNotFoundError(
            f"Dataset '{self.config.dataset_slug}' não encontrado em {self.config.kaggle_input_root}"
        )

    @staticmethod
    def _run_cmd(cmd: Sequence[str], cwd: Path | None = None) -> None:
        """Executa comando de shell validando código de retorno."""
        LOGGER.info(">>> %s", " ".join(cmd))
        subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)

    def _setup_environment(self) -> None:
        """Clona o repositório na branch exata, instala nnU-Net v2 e exporta variáveis."""
        if self.repo_dir.exists():
            shutil.rmtree(self.repo_dir)
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
        self._run_cmd([sys.executable, "-m", "pip", "install", "-q", "nnunetv2", "nibabel"])

        for folder in (self.nnunet_raw, self.nnunet_preprocessed, self.nnunet_results, self.preds_dir, self.export_dir):
            folder.mkdir(parents=True, exist_ok=True)

        os.environ["nnUNet_raw"] = str(self.nnunet_raw)
        os.environ["nnUNet_preprocessed"] = str(self.nnunet_preprocessed)
        os.environ["nnUNet_results"] = str(self.nnunet_results)

    def _package_nifti_predictions(self, images_ts: Path, labels_ts: Path) -> Path:
        """Organiza LGE + GT + PRED (`int16`, affine preservada) por paciente e gera ZIP."""
        bundle_dir = self.config.scratch_dir / "niftis_predicoes_nnunet_2d"
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        for pred_file in sorted(self.preds_dir.glob("*.nii.gz")):
            pid = pred_file.name[:-7]
            p_dir = bundle_dir / pid
            p_dir.mkdir(parents=True, exist_ok=True)

            img_src = images_ts / f"{pid}_0000.nii.gz"
            gt_src = labels_ts / f"{pid}.nii.gz"
            if img_src.exists():
                shutil.copy2(img_src, p_dir / f"{pid}_LGE.nii.gz")
            if gt_src.exists():
                shutil.copy2(gt_src, p_dir / f"{pid}_GT.nii.gz")

            # Salva predição em int16 com o cabeçalho físico exato da imagem de referência
            ref_nii = nib.load(str(img_src if img_src.exists() else pred_file))
            pred_arr = np.rint(np.squeeze(nib.load(str(pred_file)).get_fdata())).astype(np.int16)
            out_nii = nib.Nifti1Image(pred_arr, ref_nii.affine)
            out_nii.header["pixdim"] = ref_nii.header["pixdim"]
            out_nii.set_data_dtype(np.int16)
            nib.save(out_nii, str(p_dir / f"{pid}_PRED.nii.gz"))

        zip_base = self.export_dir / "niftis_predicoes_nnunet_2d"
        zip_path = Path(shutil.make_archive(str(zip_base), "zip", str(bundle_dir)))
        LOGGER.info("ZIP de predições NIfTI pronto: %s", zip_path)
        return zip_path

    def run(self) -> None:
        """Executa preparação, pré-processamento, treino 2D, inferência e avaliação."""
        self._setup_environment()
        input_dataset_dir = self._resolve_input_dataset_dir()
        dataset_dir = self.nnunet_raw / self.config.dataset_name
        images_ts = dataset_dir / "imagesTs"
        labels_ts = dataset_dir / "labelsTs"

        # 1. Prepara Dataset501 com split de teste fixo (SEED=42)
        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.data.prepare_nnunet",
                "--input-dir",
                str(input_dataset_dir),
                "--nnunet-raw",
                str(self.nnunet_raw),
            ],
            cwd=self.repo_dir,
        )

        # 2. Planejamento e pré-processamento com verificação estrita de integridade
        self._run_cmd(
            [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                str(self.config.dataset_id),
                "--verify_dataset_integrity",
                "-c",
                self.config.configuration,
            ]
        )

        # 3. Injeta splits_final.json garantindo que Fold 0 espelha exatamente SEED=42
        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.data.prepare_nnunet",
                "--input-dir",
                str(input_dataset_dir),
                "--nnunet-raw",
                str(self.nnunet_raw),
                "--nnunet-preprocessed",
                str(self.nnunet_preprocessed),
                "--inject-splits-only",
            ],
            cwd=self.repo_dir,
        )

        # 4. Treinamento 2D com --npz para preservar probabilidades softmax
        self._run_cmd(
            [
                "nnUNetv2_train",
                str(self.config.dataset_id),
                self.config.configuration,
                str(self.config.fold),
                "-tr",
                self.config.trainer,
                "--npz",
            ]
        )

        # 5. Inferência no conjunto de teste inédito (N=7)
        self._run_cmd(
            [
                "nnUNetv2_predict",
                "-i",
                str(images_ts),
                "-o",
                str(self.preds_dir),
                "-d",
                str(self.config.dataset_id),
                "-c",
                self.config.configuration,
                "-f",
                str(self.config.fold),
                "-tr",
                self.config.trainer,
            ]
        )

        # 6. Avaliação volumétrica 3D e geração de painéis comparativos
        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.evaluation.metrics_3d",
                "--gt-dir",
                str(labels_ts),
                "--pred-dir",
                str(self.preds_dir),
                "--output-dir",
                str(self.export_dir),
            ],
            cwd=self.repo_dir,
        )
        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.evaluation.visualize",
                "--image-dir",
                str(images_ts),
                "--gt-dir",
                str(labels_ts),
                "--pred-dir",
                str(self.preds_dir),
                "--output-dir",
                str(self.export_dir / "figures"),
            ],
            cwd=self.repo_dir,
        )

        # 7. Exporta planos da nnU-Net e pacote NIfTI para inspeção no 3D Slicer
        plans_src = self.nnunet_preprocessed / self.config.dataset_name / "nnUNetPlans.json"
        if plans_src.exists():
            shutil.copy2(plans_src, self.export_dir / "nnUNetPlans.json")
        self._package_nifti_predictions(images_ts, labels_ts)
        LOGGER.info("Job dlcs-nnunet-2d finalizado com sucesso. Artefatos em %s", self.export_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    KaggleNnUNet2DRunner().run()
