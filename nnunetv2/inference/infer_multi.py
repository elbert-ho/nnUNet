from nnunetv2.inference.predict_from_raw_data_2 import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join

import torch 
import einops as E
from pylot.util.torchutils import to_device

import buttermilk
from buttermilk.metrics.segmentation import seg_dice, compute_stats

from pylot.util.ioutil import autosave
import numpy as np
import os
import glob
import nibabel as nib
import pandas as pd
from pathlib import Path

def main():
    dataset_ids = [100, 670, 671, 672, 673, 760]
    dataset_names = ["ProstateTk1", "STAREm", "ProstateTk2", "BrainGrowth", "QubiqKidney", "LIDC_IDRIv3"]
    for dataset_id, dataset_name in zip(dataset_ids, dataset_names):

        if dataset_id == 100 or dataset_id == 671 or dataset_id == 760:
            config_file = "/data/ddmg/buttermilk/users/eho29/code/buttermilk/configs/megamedical_nnunet_6stage_framework.yml"
        else:
            config_file = "/data/ddmg/buttermilk/users/eho29/code/buttermilk/configs/megamedical_nnunet_5stage_framework.yml"

        model_folder = f"/data/ddmg/buttermilk/users/eho29/data/nnUNet_results/Dataset{dataset_id}_{dataset_name}/nnUNetTrainer__nnUNetPlans__2d"  # folder containing fold_0, fold_1, ... etc
        input_folder = f"/data/ddmg/buttermilk/users/eho29/data/{dataset_name}_Val/imagesVal"       # where your .nii.gz live (properly named _0000 etc)
        output_folder = f"/data/ddmg/buttermilk/users/eho29/data/{dataset_name}_Val/predictions"   # will be created if needed
        pred_root = output_folder
        gt_root = f"/data/ddmg/buttermilk/users/eho29/data/{dataset_name}_Val/labelsVal"
        df_path = f"/data/ddmg/buttermilk/users/eho29/data/{dataset_name}_Val"
        df_path = Path(df_path)

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,        # if you want TTA
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        # use_folds can be (0,1,2,3,4) or whatever you trained
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=("0",),            # or multiple folds if you like
            checkpoint_name='checkpoint_best.pth',
            config_file=config_file
        )

        predictor.predict_from_files(
            input_folder,
            output_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )

        # PART 2 = LOADING IN AND CREATING DATAFRAME
        def load_nifti_as_tensor(path, dtype=torch.float32):
            img = nib.load(path)
            data = img.get_fdata()
            # for segmentations: integer labels
            if dtype is torch.long:
                data = data.astype(np.int64)

            return torch.from_numpy(data).to(dtype)

        gt_paths = sorted(glob.glob(os.path.join(gt_root, "*.pt")))
        K = 8

        rows = []
        rows_dists = []

        for (i, gt_path) in enumerate(gt_paths):
            case_id = os.path.basename(gt_path).replace(".pt", "")
            gt = torch.load(gt_path, map_location="cpu").to(torch.float32)
            # gt = load_nifti_as_tensor(gt_path, dtype=torch.long)


            # print(gt.dtype)
            # exit()

            preds_k = []
            for k in range(K):
                pred_path = os.path.join(pred_root, f"{case_id}_k{k}.nii.gz")
                if not os.path.isfile(pred_path):
                    raise FileNotFoundError(f"Missing prediction {pred_path}")
                preds_k.append(load_nifti_as_tensor(pred_path, dtype=torch.float32))

            preds_k = torch.stack(preds_k, dim=0)

            # print(preds_k.shape)
            # print(gt.shape)

            y_pred = preds_k.unsqueeze(1)
            y_pred = y_pred.unsqueeze(0)
            y = gt.unsqueeze(0)

            Ki = K

            n_labels = y.shape[1]
            for label in range(n_labels):
                spec_label = y[:, label, ...]
                spec_label = spec_label[:, None, ...]

                spec_label = E.repeat(spec_label[:, None, ...], "B 1 C H W -> B K C H W", K=Ki)
                assert y.shape[0] == 1        
                assert spec_label.shape[0] == 1 

                pred_mean = torch.mean(y_pred, dim=1, keepdim=True)

                dice = seg_dice(
                    pred_mean, 
                    spec_label,
                    from_logits=False,
                    dim=1
                    ).item()

                hard_dice = seg_dice(
                    pred_mean>0.5, 
                    spec_label>0.5,
                    from_logits=False,
                    dim=1
                ).item()


                bdice = seg_dice(
                    y_pred, spec_label, from_logits=False, dim=1,
                ).item()

                bhard_dice = seg_dice(y_pred>0.5, spec_label>0.5, 
                                                            from_logits=False, dim=1).item()


                rows.append(
                    {
                        "phase": "val",
                        "subject": i,
                        "ensemble_score": dice,
                        "best_soft_dice": bdice,
                        "ensemble_hard_dice_score": hard_dice,
                        "best_hard_dice": bhard_dice,
                        "rater": label+1,
                        'dataset': 'MegaMedical',
                    }
                )

            y = y.unsqueeze(2)
            for y_sample, y_pred_sample in zip(y, y_pred):
                
                ged, sdiv, hm, best_dice, _ = compute_stats(y_pred_sample, y_sample)
                ged, sdiv, hm, best_dice = ged.item(), sdiv.item(), hm.item(), best_dice.item()

                rows_dists.append(
                    {
                        "phase": "val",
                        "subject": i,
                        "dataset": "Megamedical",
                        "GED": ged,
                        "Diversity_Score": sdiv,
                        "Hungarian_Matching": hm,
                        "Best_Dice": best_dice
                    }
                )

        df = pd.DataFrame.from_records(rows)
        autosave(df, df_path / "inference" / "breakdown-val_id.parquet")

        if rows_dists is not None:
            df_dists = pd.DataFrame.from_records(rows_dists)
            autosave(df_dists, df_path / "inference" / "breakdown-dists-val_id.parquet")
    
        print(f"Finished processing {dataset_name}")

if __name__ == "__main__":
    main()
