import sys
import os
from os.path import join
import numpy as np
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
from time import time
import argparse

from deepmri import dsutils  # noqa: E402

def main():
    parser = argparse.ArgumentParser(
        description='Trains and evaluates a RF classifier for given features.',
        add_help=False)

    parser.add_argument('indir',
                        help='Folder containing all required input files')
    parser.add_argument('features',
                        help='Name of the feature set that should be used')
    parser.add_argument('ffile',
                        help='Name of the file containing the features')
    args = parser.parse_args()

    data_dir = join(args.indir, "data")
    features_name = args.features
    features_file = args.ffile
    
    # ----------------------------------------------Settings----------------------------------------------
    subj_id = "784565"

    labels = ["Other", "CG", "CST", "FX", "CC"]
    tract_masks_pth = join(data_dir, subj_id, "Diffusion/tract_masks/tract_masks.nii.gz")
    train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]

    #labels = ["Other", "IFO_left", "IFO_right", "ILF_left", "ILF_right", "SLF_left", "SLF_right"]
    #tract_masks_pth = join(data_dir, subj_id, "Diffusion/tract_masks/tract_masks_2.nii.gz")
    #train_slices = [('sagittal', 44), ('sagittal', 102), ('coronal', 65)]

    features_path = join(data_dir, subj_id, "Diffusion", "features", features_file)

    min_samples_leaf = 8
    gen_probs = True

    outputs_dir = join(data_dir, subj_id, "Diffusion", "outputs")
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    results_pth = join(outputs_dir, features_name + "_dice_scores.npz")
    probs_path = join(outputs_dir, features_name + "_probs_coords.npz")

    # ---------------------------------------------Load Data----------------------------------------------
    print("Loading Data".center(100, "-"))

    tract_masks = nib.load(tract_masks_pth).get_fdata()
    tract_masks = tract_masks[:, :, :, 1:]  # remove background class

    if features_path.endswith(".npz"):
        features = np.load(features_path)["data"]
    else:
        features = nib.load(features_path).get_fdata()

    # features = features[:, :, :, 1:]

    print("features file: {}".format(features_file))
    print("features name: {}, shape: {}".format(features_name, features.shape))

    # ---------------------------------------------Train Set----------------------------------------------
    print("Preparing the training set".center(100, '-'))

    train_masks = dsutils.create_data_masks(tract_masks, train_slices, labels)
    np.savez(join(outputs_dir, "trainset_labels.npz"), data=train_masks)

    X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(features,
                                                                           train_masks,
                                                                           labels=labels,
                                                                           multi_label=True)
    print("X_train shape: {}, y_train shape: {}".format(X_train.shape, y_train.shape))

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
    print("Train time: {:.2f} seconds.".format(time() - start))
    # ----------------------------------------------Test Set----------------------------------------------

    print("Testset".center(100, "-"))
    X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(features,
                                                                        tract_masks,
                                                                        labels=labels,
                                                                        multi_label=True)
    print("X_test shape: {}, y_test shape: {}".format(X_test.shape, y_test.shape))

    # ---------------------------------------Evaluation on test set---------------------------------------
    start = time()
    test_preds = clf.predict(X_test)
    scores = dsutils.get_scores(y_test, test_preds, labels)
    print("Inference time: {:.2f} seconds.".format(time() - start))
    for k in sorted(scores.keys()):
        print(f"{k}: {scores[k]:.4f}")

    if gen_probs:
        test_probs = clf.predict_proba(X_test)
        test_probs = np.array(test_probs)[:, :, 1]
        print(test_probs.shape, y_test.T.shape)
        np.savez(probs_path, probs=test_probs, coords=test_coords)
        # np.savez(probs_path + ".gt", probs=y_test.T, coords=test_coords)

if __name__ == "__main__":
    main()
