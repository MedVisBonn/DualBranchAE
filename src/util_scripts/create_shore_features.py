import nibabel as nib
from dipy.reconst.shore import ShoreModel
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
import os
from os.path import join
from time import time
import numpy as np
import argparse

def main():
    script_start = time()

    parser = argparse.ArgumentParser(
        description='Creates SHORE features from dMRI data.',
        add_help=False)

    parser.add_argument('indir',
                        help='Folder containing all required input files')
    args = parser.parse_args()
    exp_dir = args.indir
    
    # ----------------------------------------------Settings----------------------------------------------
    start = time()
    print("Settings".center(100, "-"))
    radial_order = 4

    subj_ids = ["784565"]
    for idx, subj_id in enumerate(subj_ids, 1):
        print("Subject {}/{}: {}".format(subj_id, len(subj_ids), subj_id))
        data_dir = join(exp_dir, "data", subj_id, "Diffusion")

        features_dir = join(data_dir, "features")
        if not os.path.exists(features_dir):
            os.mkdir(features_dir)

        save_pth = join(features_dir, "shore_coefficients_radial_order_{}.npz".format(radial_order))
        data_pth = join(data_dir, "data.nii.gz")
        mask_pth = join(data_dir, "nodif_brain_mask.nii.gz")
        fbval = join(data_dir, "bvals")
        fbvec = join(data_dir, "bvecs")

        # --------------------------------------------Data Loading--------------------------------------------
        start = time()
        print("Data Loading".center(100, "-"))

        # make gradient table
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)

        # load dmri data
        data = nib.load(data_pth).get_fdata()

        # load mask
        mask = nib.load(mask_pth).get_fdata()

        print("Data shape: ", data.shape)
        print("Mask shape: ", mask.shape)

        print("Section time: {:.2f} seconds.".format(time() - start))
        # ----------------------------------------Fitting SHORE Model-----------------------------------------
        start = time()
        print("Fitting SHORE model with radial_order={}".format(radial_order).center(100, "-"))

        asm = ShoreModel(gtab, radial_order=radial_order)
        asmfit = asm.fit(data, mask)
        shore_coeff = asmfit.shore_coeff
        print("SHORE coefficients shape: ", shore_coeff.shape)

        # replace nan with 0
        print("Replacing nan with 0. This happens because of inaccurate mask.")
        shore_coeff = np.nan_to_num(shore_coeff, 0)

        print("Section time: {:.2f} seconds.".format(time() - start))
        # ------------------------------------------Saving features-------------------------------------------
        start = time()
        print("Saving features".center(100, "-"))
        np.savez(save_pth, data=shore_coeff)

        print("Section time: {:.2f} seconds.".format(time() - start))
        # ----------------------------------------------------------------------------------------------------
    print("Total script time: {:.2f} seconds.".format(time()-script_start))

if __name__ == "__main__":
    main()
