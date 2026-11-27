"""Runner fino para Exp 04B: nnU-Net v2 2D ResEncM (5 classes — Anatomia + Scar) no Kaggle.

Treina no `Dataset501_CMRMULTI_LGE` (`0=Bg, 1=VE, 2=MioSaudavel, 3=Scar, 4=VD`),
Fold 0 (`SEED=42`), com `nnUNetPlannerResEncM` (Isensee et al., MICCAI 2024) e
`nnUNetTrainer_250epochs` otimizado para Tesla T4 (`nnUNet_n_proc_DA=4`, `compile=false`).
Avalia no CMR-MULTI Test (`N=7`) e na coorte clínica privada HU-UFJF (`N=10` + `patient_20`),
exportando NIfTIs (`int16`), métricas clínicas + MICCAI 2026 (brutas e pós-processadas),
figuras e o pacote reutilizável de pesos (`nnunet_model_bundle_5c.zip`).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import nibabel as nib
import numpy as np

LOGGER = logging.getLogger("dlcs_nnunet_2d_5c_runner")


@dataclass(frozen=True)
class KaggleNnUNetJobConfig:
    """Parâmetros imutáveis do job `dlcs-nnunet-2d-5c`."""

    repo_url: str = "https://github.com/robertgvds/DL-Cardiac-Segmentation-Research.git"
    branch: str = "exp/nnunet-2d-cmr-multi"
    task_mode: str = "full_5c"
    dataset_id: int = 501
    dataset_name: str = "Dataset501_CMRMULTI_LGE"
    cmr_slug: str = "cmr-multi-lge-sax"
    hu_slug: str = "hu-ufjf-lge-sax"
    kaggle_owner: str = "robertgvds"
    planner: str = "nnUNetPlannerResEncM"
    plans_name: str = "nnUNetResEncUNetMPlans"
    configuration: str = "2d"
    fold: int = 0
    trainer: str = "nnUNetTrainer_250epochs"
    num_da_workers: int = 4
    kaggle_input_root: Path = Path("/kaggle/input")
    scratch_dir: Path = Path("/tmp/dlcs_nnunet_5c")
    working_dir: Path = Path("/kaggle/working")


class KaggleNnUNetResEncRunner:
    """Orquestrador fino para treino, avaliação dupla (CMR-MULTI + HU-UFJF) e exportação de pesos."""

    def __init__(self, config: KaggleNnUNetJobConfig | None = None) -> None:
        self.config = config or KaggleNnUNetJobConfig()
        self.repo_dir = self.config.scratch_dir / "repo"
        self.nnunet_raw = self.config.scratch_dir / "nnUNet_raw"
        self.nnunet_preprocessed = self.config.scratch_dir / "nnUNet_preprocessed"
        self.nnunet_results = self.config.scratch_dir / "nnUNet_results"
        self.hu_work_dir = self.config.scratch_dir / "hu_prepared"
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
    def _run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
        """Executa comando de shell validando código de retorno."""
        LOGGER.info(">>> %s", " ".join(cmd))
        subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)

    def _setup_environment(self) -> None:
        """Clona o repositório na branch exata, instala nnU-Net v2 e configura env vars."""
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

        for folder in (self.nnunet_raw, self.nnunet_preprocessed, self.nnunet_results, self.hu_work_dir, self.export_dir):
            folder.mkdir(parents=True, exist_ok=True)

        os.environ["nnUNet_raw"] = str(self.nnunet_raw)
        os.environ["nnUNet_preprocessed"] = str(self.nnunet_preprocessed)
        os.environ["nnUNet_results"] = str(self.nnunet_results)
        # Evita thrashing nas 4 vCPUs do Kaggle e overhead de torch.compile na Tesla T4 (sm_75)
        os.environ["nnUNet_n_proc_DA"] = str(self.config.num_da_workers)
        os.environ["nnUNet_compile"] = "false"
        subprocess.run(["nvidia-smi"], check=False)

    def _resolve_active_plans_name(self) -> str:
        """Confirma o identificador exato do arquivo `*Plans.json` gerado pelo planner."""
        preproc_ds = self.nnunet_preprocessed / self.config.dataset_name
        expected = preproc_ds / f"{self.config.plans_name}.json"
        if expected.is_file():
            return self.config.plans_name
        for cand in sorted(preproc_ds.glob("*Plans.json")):
            LOGGER.info("Usando plans detectado automaticamente: %s", cand.stem)
            return cand.stem
        return "nnUNetPlans"

    def _package_cohort_niftis(
        self,
        cohort_key: str,
        images_dir: Path,
        labels_dir: Optional[Path],
        pred_dir: Path,
        out_dir: Path,
    ) -> Path:
        """Empacota LGE + GT + PRED (`int16`, affine/pixdim preservados) em ZIP para 3D Slicer."""
        bundle_dir = self.config.scratch_dir / f"bundle_{cohort_key}"
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        for pred_file in sorted(pred_dir.glob("*.nii.gz")):
            pid = pred_file.name[:-7]
            p_dir = bundle_dir / pid
            p_dir.mkdir(parents=True, exist_ok=True)

            img_src = images_dir / f"{pid}_0000.nii.gz"
            gt_src = (labels_dir / f"{pid}.nii.gz") if labels_dir is not None else None
            if img_src.exists():
                shutil.copy2(img_src, p_dir / f"{pid}_LGE.nii.gz")
            if gt_src is not None and gt_src.exists():
                shutil.copy2(gt_src, p_dir / f"{pid}_GT_{self.config.task_mode}.nii.gz")

            ref_nii = nib.load(str(img_src if img_src.exists() else pred_file))
            pred_arr = np.rint(np.squeeze(nib.load(str(pred_file)).get_fdata())).astype(np.int16)
            out_nii = nib.Nifti1Image(pred_arr, ref_nii.affine)
            out_nii.header["pixdim"] = ref_nii.header["pixdim"]
            out_nii.set_data_dtype(np.int16)
            nib.save(out_nii, str(p_dir / f"{pid}_PRED_{self.config.task_mode}.nii.gz"))

        zip_base = out_dir / f"niftis_predicoes_{cohort_key}_{self.config.task_mode}"
        zip_path = Path(shutil.make_archive(str(zip_base), "zip", str(bundle_dir)))
        shutil.rmtree(bundle_dir, ignore_errors=True)
        return zip_path

    def _export_reusable_model_bundle(self, plans_name: str) -> Path:
        """Empacota `dataset.json`, `plans.json`, `progress.png` e pesos `.pth` sem `.npz` pesados."""
        model_root = (
            self.nnunet_results
            / self.config.dataset_name
            / f"{self.config.trainer}__{plans_name}__{self.config.configuration}"
        )
        fold_dir = model_root / f"fold_{self.config.fold}"

        for name in ("plans.json", "dataset.json"):
            src = model_root / name
            if src.is_file():
                shutil.copy2(src, self.export_dir / name)
        if (fold_dir / "progress.png").is_file():
            shutil.copy2(fold_dir / "progress.png", self.export_dir / "progress.png")

        bundle_zip = self.export_dir / f"nnunet_model_bundle_{self.config.task_mode}.zip"
        with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(model_root.rglob("*")):
                if not item.is_file():
                    continue
                if item.suffix == ".npz":
                    continue
                if "validation" in item.parts and item.suffix in (".gz", ".nii"):
                    continue
                rel = item.relative_to(self.nnunet_results)
                zf.write(item, arcname=str(rel))

        LOGGER.info("Pacote reutilizável de pesos nnU-Net v2 salvo em %s", bundle_zip)
        return bundle_zip

    def _evaluate_cohort(
        self,
        cohort_key: str,
        cohort_display: str,
        images_dir: Path,
        labels_dir: Path,
        raw_pred_dir: Path,
        unlabeled_ids: Sequence[str] = (),
    ) -> None:
        """Avalia predições brutas e pós-processadas (LCC 3D), gera figuras e ZIP NIfTI."""
        cohort_out = self.export_dir / cohort_key
        raw_eval_dir = cohort_out / "raw_eval"
        post_pred_dir = self.config.scratch_dir / f"preds_{cohort_key}_postproc"
        cohort_out.mkdir(parents=True, exist_ok=True)

        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.postprocessing.cardiac_postproc",
                "--pred-dir",
                str(raw_pred_dir),
                "--output-dir",
                str(post_pred_dir),
                "--ref-image-dir",
                str(images_dir),
                "--task-mode",
                self.config.task_mode,
            ],
            cwd=self.repo_dir,
        )

        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.evaluation.metrics_3d",
                "--gt-dir",
                str(labels_dir),
                "--pred-dir",
                str(raw_pred_dir),
                "--output-dir",
                str(raw_eval_dir),
                "--task-mode",
                self.config.task_mode,
                "--cohort-name",
                f"{cohort_display} (Raw)",
            ],
            cwd=self.repo_dir,
        )
        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.evaluation.metrics_3d",
                "--gt-dir",
                str(labels_dir),
                "--pred-dir",
                str(post_pred_dir),
                "--output-dir",
                str(cohort_out),
                "--task-mode",
                self.config.task_mode,
                "--cohort-name",
                cohort_display,
            ],
            cwd=self.repo_dir,
        )

        model_label = f"nnU-Net v2 ResEncM ({self.config.task_mode})"
        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.evaluation.visualize",
                "--image-dir",
                str(images_dir),
                "--gt-dir",
                str(labels_dir),
                "--pred-dir",
                str(post_pred_dir),
                "--output-dir",
                str(cohort_out / "figures"),
                "--task-mode",
                self.config.task_mode,
                "--model-label",
                model_label,
            ],
            cwd=self.repo_dir,
        )
        if unlabeled_ids:
            sys.path.insert(0, str(self.repo_dir))
            from src.data.cardiac_dataset import TaskMode
            from src.evaluation.visualize import SegmentationPanelVisualizer

            vis = SegmentationPanelVisualizer(
                image_dir=images_dir,
                gt_dir=None,
                pred_dir=post_pred_dir,
                output_dir=cohort_out / "figures",
                task_mode=TaskMode(self.config.task_mode),
                model_label=model_label,
            )
            for pid in unlabeled_ids:
                img_p = images_dir / f"{pid}_0000.nii.gz"
                pred_p = post_pred_dir / f"{pid}.nii.gz"
                if img_p.is_file() and pred_p.is_file():
                    vis.render_unlabeled_case(pid, img_p, pred_p)

        self._package_cohort_niftis(cohort_key, images_dir, labels_dir, post_pred_dir, cohort_out)

    def run(self) -> None:
        """Executa preparação, planejamento ResEncM, treino 2D, inferência dupla e exportação."""
        self._setup_environment()
        cmr_dir = self._resolve_input_dataset(self.config.cmr_slug)
        hu_dir = self._resolve_input_dataset(self.config.hu_slug)

        dataset_dir = self.nnunet_raw / self.config.dataset_name
        images_ts = dataset_dir / "imagesTs"
        labels_ts = dataset_dir / "labelsTs"
        preds_cmr_raw = self.config.scratch_dir / "preds_cmr_raw"
        preds_hu_raw = self.config.scratch_dir / "preds_hu_raw"
        preds_cmr_raw.mkdir(parents=True, exist_ok=True)
        preds_hu_raw.mkdir(parents=True, exist_ok=True)

        # 1. Converte CMR-MULTI (5c) + HU-UFJF padronizado
        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.data.prepare_nnunet",
                "--input-dir",
                str(cmr_dir),
                "--nnunet-raw",
                str(self.nnunet_raw),
                "--task-mode",
                self.config.task_mode,
                "--hu-input-dir",
                str(hu_dir),
                "--hu-output-dir",
                str(self.hu_work_dir),
            ],
            cwd=self.repo_dir,
        )

        # 2. Planejamento ResEncM (MICCAI 2024) e pré-processamento 2D
        self._run_cmd(
            [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                str(self.config.dataset_id),
                "-pl",
                self.config.planner,
                "--verify_dataset_integrity",
                "-c",
                self.config.configuration,
            ]
        )
        active_plans = self._resolve_active_plans_name()

        # 3. Injeta splits_final.json (Fold 0 espelhando exatamente SEED=42)
        self._run_cmd(
            [
                sys.executable,
                "-m",
                "src.data.prepare_nnunet",
                "--input-dir",
                str(cmr_dir),
                "--nnunet-raw",
                str(self.nnunet_raw),
                "--task-mode",
                self.config.task_mode,
                "--nnunet-preprocessed",
                str(self.nnunet_preprocessed),
                "--inject-splits-only",
            ],
            cwd=self.repo_dir,
        )

        # 4. Treinamento 2D ResEncM (250 épocas, --npz para probabilidades)
        self._run_cmd(
            [
                "nnUNetv2_train",
                str(self.config.dataset_id),
                self.config.configuration,
                str(self.config.fold),
                "-tr",
                self.config.trainer,
                "-p",
                active_plans,
                "--npz",
            ]
        )

        # 5. Inferência no CMR-MULTI Test (N=7) e na coorte externa HU-UFJF (N=10 + patient_20)
        for in_dir, out_dir in (
            (images_ts, preds_cmr_raw),
            (self.hu_work_dir / "imagesHu", preds_hu_raw),
        ):
            self._run_cmd(
                [
                    "nnUNetv2_predict",
                    "-i",
                    str(in_dir),
                    "-o",
                    str(out_dir),
                    "-d",
                    str(self.config.dataset_id),
                    "-c",
                    self.config.configuration,
                    "-f",
                    str(self.config.fold),
                    "-tr",
                    self.config.trainer,
                    "-p",
                    active_plans,
                ]
            )

        # 6. Avaliação completa (CMR-MULTI Test + HU-UFJF External)
        self._evaluate_cohort(
            cohort_key="cmr_multi_test",
            cohort_display="CMR-MULTI Test",
            images_dir=images_ts,
            labels_dir=labels_ts,
            raw_pred_dir=preds_cmr_raw,
        )
        hu_manifest = json.loads((self.hu_work_dir / "hu_manifest.json").read_text(encoding="utf-8"))
        self._evaluate_cohort(
            cohort_key="hu_ufjf_external",
            cohort_display="HU-UFJF Private Cohort",
            images_dir=self.hu_work_dir / "imagesHu",
            labels_dir=self.hu_work_dir / "labelsHu",
            raw_pred_dir=preds_hu_raw,
            unlabeled_ids=hu_manifest.get("unlabeled_ids", []),
        )

        # 7. Exporta planos, progresso e pacote reutilizável de pesos
        plans_src = self.nnunet_preprocessed / self.config.dataset_name / f"{active_plans}.json"
        if plans_src.is_file():
            shutil.copy2(plans_src, self.export_dir / f"{active_plans}.json")
        self._export_reusable_model_bundle(active_plans)
        LOGGER.info("Job dlcs-nnunet-2d-5c concluído com sucesso! Saídas em %s", self.export_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    KaggleNnUNetResEncRunner().run()
