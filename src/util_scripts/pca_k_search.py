from os.path import join
from pathlib import Path
import numpy as np
import nibabel as nib
import argparse
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import sys
from time import time

from deepmri import dsutils  # noqa: E402

def main():
    start = time()

    parser = argparse.ArgumentParser(
        description='Searches for the best number of PCA components.',
        add_help=False)

    parser.add_argument('indir',
                        help='Folder containing all required input files')
    args = parser.parse_args()
    DATA_DIR = join(args.indir, "data")
    
    # settings
    SUBJ_ID = "784565"
    DATA_PATH = join(DATA_DIR, SUBJ_ID, "Diffusion/data.nii.gz")
    MASK_PATH = join(DATA_DIR, SUBJ_ID, "Diffusion/nodif_brain_mask.nii.gz")

    # set 1
    TRACT_MASKS_PTH = join(DATA_DIR, SUBJ_ID, "Diffusion/tract_masks", "tract_masks.nii.gz")
    LABELS = ["Other", "CG", "CST", "FX", "CC"]
    train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
    set_id = "1"
    #
    # TRACT_MASKS_PTH = join(DATA_DIR, SUBJ_ID, "Diffusion/tract_masks", "tract_masks_2.nii.gz")
    # LABELS = ["Other", "IFO_left", "IFO_right", "ILF_left", "ILF_right", "SLF_left", "SLF_right"]
    # train_slices = [('sagittal', 44), ('sagittal', 102), ('coronal', 65)]
    # set_id = "2"

    # load data
    DATA = nib.load(DATA_PATH).get_fdata()
    MASK = nib.load(MASK_PATH).get_fdata()
    TRACT_MASKS = nib.load(TRACT_MASKS_PTH).get_fdata()
    TRACT_MASKS = TRACT_MASKS[:, :, :, 1:]  # remove background class

    ncs = list(range(1, 51))
    norms = [False, True]

    for norm in norms:
        stats = {
            "n_components": [],
            "scores": []
        }
        sv = f"pca_stats_set_{set_id}.npz"
        if norm:
            sv = "norm_" + sv
        
        SAVE_PATH = join(DATA_DIR, SUBJ_ID, f"Diffusion/outputs/{sv}")
        Path(join(DATA_DIR, SUBJ_ID, f"Diffusion/outputs")).mkdir(parents=True, exist_ok=True)
        for nc in ncs:
            print("n_components = {}".format(nc).center(100, "-"))
            features, _, _ = dsutils.make_pca_volume(DATA, MASK, n_components=nc, normalize=norm)
            train_masks = dsutils.create_data_masks(TRACT_MASKS, train_slices, LABELS)
            X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(features,
                                                                                   train_masks,
                                                                                   labels=LABELS,
                                                                                   multi_label=True)
            X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(features,
                                                                                TRACT_MASKS,
                                                                                labels=LABELS,
                                                                                multi_label=True)
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            clf = RandomForestClassifier(n_estimators=100,
                                         bootstrap=True,
                                         oob_score=True,
                                         random_state=0,
                                         n_jobs=-1,
                                         max_features='auto',
                                         class_weight='balanced',
                                         max_depth=None,
                                         min_samples_leaf=8)
            clf.fit(X_train, y_train)
            test_preds = clf.predict(X_test)
            test_f1_macro = sklearn.metrics.f1_score(y_test[:, 1:], test_preds[:, 1:], average='macro')
            stats["n_components"].append(nc)
            stats["scores"].append(test_f1_macro)
            print(test_f1_macro)

        np.savez(SAVE_PATH, n_components=stats["n_components"], scores=stats["scores"])

    print(f"Run time: {time() - start}:.2f seconds")

if __name__ == "__main__":
    main()
