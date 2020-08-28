import nibabel as nib
import os
from os.path import join
from pathlib import Path
from time import time
import numpy as np
import argparse
import pickle
from deepmri import dsutils  # noqa: E402

def main():
    script_start = time()
        
    parser = argparse.ArgumentParser(
        description='Creates PCA features from dMRI data.',
        add_help=False)

    parser.add_argument('indir',
                        help='Folder containing all required input files')
    args = parser.parse_args()
    exp_dir = args.indir

    # ----------------------------------------------Settings----------------------------------------------
    start = time()
    print("Settings".center(100, "-"))

    n_components = 11
    normalize = False
    reconstruct = False
    load_pca = False
    save_pca = False
    run_clf = False

    subj_ids = ["784565"]
    # subj_ids = os.listdir(join(exp_dir, "data"))
    for idx, subj_id in enumerate(subj_ids, 1):
        print("Subject {}/{}: {}".format(idx, len(subj_ids), subj_id))
        data_dir = join(exp_dir, "data", subj_id, "Diffusion")

        if load_pca:
            print("Loading pca model.")
            with open("/home/agajan/experiment_DiffusionMRI/saved_models/pca.model", "rb") as f:
                pca = pickle.load(f)
        else:
            pca = None

        features_dir = join(data_dir, "features")
        if not os.path.exists(features_dir):
            os.mkdir(features_dir)

        ft_name = "pca_features_k_{}.npz".format(n_components)

        # pca is recomputed
        if not load_pca:
            ft_name = "recomputed_" + ft_name

        save_pth = join(features_dir, ft_name)

        recons_dir = join(data_dir, "reconstructions")

        if not os.path.exists(recons_dir):
            os.mkdir(recons_dir)

        recon_save_pth = join(recons_dir, "pca_{}_recons_k_{}.nii.gz".format(subj_id, n_components))
        data_pth = join(data_dir, "data.nii.gz")
        mask_pth = join(data_dir, "nodif_brain_mask.nii.gz")

        print("Section time: {:.2f} seconds.".format(time() - start))
        # --------------------------------------------Data Loading--------------------------------------------
        start = time()
        print("Data Loading".center(100, "-"))

        # load dmri data
        dmri = nib.load(data_pth)
        data = dmri.get_fdata()

        # load mask
        mask = nib.load(mask_pth).get_fdata()

        print("Data shape: ", data.shape)
        print("Mask shape: ", mask.shape)

        print("Section time: {:.2f} seconds.".format(time() - start))
        # --------------------------------------------Fitting PCA---------------------------------------------
        start = time()
        print("Fitting PCA with n_components={}".format(n_components).center(100, "-"))

        features_volume, pca, recons = dsutils.make_pca_volume(data, mask,
                                                               pca=pca, n_components=n_components,
                                                               normalize=normalize, reconstruct=reconstruct)
        if save_pca:
            Path(f"{exp_dir}saved_models").mkdir(parents=True, exist_ok=True)
            with open(f"{exp_dir}saved_models/pca.model", "wb") as f:
                pickle.dump(pca, f)

        print("Section time: {:.2f} seconds.".format(time() - start))
        # ------------------------------------------Saving features-------------------------------------------
        start = time()
        print("Saving features".center(100, "-"))

        np.savez(save_pth, data=features_volume)

        print("Section time: {:.2f} seconds.".format(time() - start))
        # -------------------------------------------Reconstruction-------------------------------------------
        if reconstruct:
            start = time()
            print("Reconstruction".center(100, "-"))

            nib.save(nib.Nifti1Image(recons, dmri.affine, dmri.header), recon_save_pth)

            print("Section time: {:.2f} seconds.".format(time() - start))
        # -------------------------------------------Run classifier-------------------------------------------
        if run_clf:
            features_name = "PCA"

            # features are re-calculated
            if not load_pca:
                features_name = features_name + "_recomputed"
            features_file = "features/" + ft_name
            print(features_file)
            os.system("python rf_classifier.py {} {} {}".format(subj_id, features_name, features_file))
        # ----------------------------------------------------------------------------------------------------
    print("Total script time: {:.2f} seconds.".format(time() - script_start))

if __name__ == "__main__":
    main()
