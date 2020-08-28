import torch
from os.path import join
from pathlib import Path
import sys
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
from time import time
import numpy as np

sys.path.append("/home/agajan/DVR-Multi-Shell/src/")
from deepmri import dsutils  # noqa: E402


def generate_features(exp_dir, subj_id, model, model_name,
                      inference_epoch, latent_dim, device,
                      dataset, conv_type, device_num=0):
    if inference_epoch != 0:
        model_path = f"{exp_dir}saved_models/{model_name}_epoch_{inference_epoch}"
        model.load_state_dict(torch.load(model_path, map_location=f"cuda:{device_num}"))
        model.eval()

    ref_img_path = join(exp_dir, "data", subj_id, "Diffusion/data.nii.gz")
    Path(join(exp_dir, "data", subj_id, "Diffusion/features")).mkdir(parents=True, exist_ok=True)
    save_pth = join(exp_dir, "data", subj_id, "Diffusion/features",
                    f"{model_name}_features_epoch_{inference_epoch}.nii.gz")

    if conv_type == 1:
        dsutils.create_voxel_features(model, (145, 174, 145, latent_dim), device, dataset, ref_img_path, save_pth)
    elif conv_type == 2:
        dsutils.create_conv_features(model, device, dataset, latent_dim, "coronal", ref_img_path, save_pth)
    elif conv_type == 3:
        dsutils.create_features_from_subvolumes(model, device, latent_dim, dataset, ref_img_path, save_pth)
    else:
        raise ValueError("Incorrect conv type")


def reconsruct(exp_dir, data_dir, subj_id, model, model_name, device, dataset,
               inference_epoch, conv_type, device_num=0):

    if inference_epoch != 0:
        model_path = f"{exp_dir}saved_models/{model_name}_epoch_{inference_epoch}"
        model.load_state_dict(torch.load(model_path, map_location=f"cuda:{device_num}"))
        model.eval()

    ref_img_pth = join(data_dir, "nodif_brain_mask.nii.gz")
    Path(join(exp_dir, "data", subj_id, "Diffusion/reconstructions")).mkdir(parents=True, exist_ok=True)
    save_pth = join(exp_dir, "data", subj_id, "Diffusion/reconstructions",
                    f"{model_name}_recons_{inference_epoch}.nii.gz")
    if conv_type == 1:
        recons = dsutils.create_voxel_reconstructions(model, dataset, (145, 174, 145, 288), device, ref_img_pth,
                                                      save_pth)
    elif conv_type == 2:
        recons = dsutils.create_conv_reconstructions(model, device, dataset, (145, 174, 145, 288), "coronal",
                                                     ref_img_pth, save_pth)
    elif conv_type == 3:
        recons = dsutils.create_reconstructions_from_subvolumes(model, device, dataset, ref_img_pth, save_pth)
    else:
        raise ValueError("Incorrect conv type")

    print("Loading data and calculating MSE.")
    dmri = nib.load(join(exp_dir, "data", subj_id, "Diffusion/data.nii.gz")).get_fdata()
    print("Original min: {}, Recons. min: {}, Original max: {}, Recons. max: {}".format(
        dmri.min(), recons.min(), dmri.max(), recons.max()
    ))

    mask = nib.load(ref_img_pth).get_fdata()

    # correct dmri for mask alignment
    dmri = dmri * mask[:, :, :, None]

    # zero out reconstructions out of brain mask
    recons = recons * mask[:, :, :, None]

    mse = np.mean((dmri - recons) ** 2)
    print("Voxel-wise MSE: {:.2f}".format(mse))
    return mse


def run_clf(min_samples_leaf, labels, data_dir, train_slices, tract_masks_path, model_name, inference_epoch, logger):

    features_path = join(data_dir, "features/{}_features_epoch_{}.nii.gz".format(model_name, inference_epoch))

    tract_masks = nib.load(tract_masks_path).get_fdata()
    tract_masks = tract_masks[:, :, :, 1:]  # remove background class

    features = nib.load(features_path).get_fdata()

    # ---------------------------------------------Train Set----------------------------------------------
    print("Train set".center(100, '-'))

    train_masks = dsutils.create_data_masks(tract_masks, train_slices, labels)
    X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(features,
                                                                           train_masks,
                                                                           labels=labels,
                                                                           multi_label=True)
    print("X_train shape: {}, y_train shape: {}".format(X_train.shape, y_train.shape))
    # ----------------------------------------------Test Set----------------------------------------------
    print("Testset".center(100, "-"))
    X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(features,
                                                                        tract_masks,
                                                                        labels=labels,
                                                                        multi_label=True)
    print("X_test shape: {}, y_test shape: {}".format(X_test.shape, y_test.shape))
    # --------------------------------------Random Forest Classifier--------------------------------------
    print("Random Forest Classifier".center(100, '-'))
    print("Fitting with min_samples_leaf={}".format(min_samples_leaf))

    start = time()
    clf = RandomForestClassifier(n_estimators=100,
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1,
                                 max_features="auto",
                                 class_weight="balanced",
                                 max_depth=None,
                                 min_samples_leaf=min_samples_leaf)
    clf.fit(X_train, y_train)
    if logger is not None:
        logger.log({
            "rf_train_time": time() - start
        })
    print("Train time: {:.2f} seconds.".format(time() - start))
    # ---------------------------------------Evaluation on test set---------------------------------------
    start = time()
    test_preds = clf.predict(X_test)
    if logger is not None:
        logger.log({
            "rf_test_time": time() - start
        })
    print("Test time: {:.2f} seconds.".format(time() - start))

    scores = dsutils.get_scores(y_test, test_preds, labels)

    return scores
