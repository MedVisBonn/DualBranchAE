import numpy as np
import sys

sys.path.append("/home/agajan/DVR-Multi-Shell/src/")
from deepmri import dsutils  # noqa: E402


def evaluate_rf(clf, X_test, y_test, labels):
    test_probs = np.array(clf.predict_proba(X_test))[:, :, 1].T
    th = 0.5
    test_preds = test_probs.copy()
    test_preds[test_preds < th] = 0
    test_preds[test_preds >= th] = 1

    scores = dsutils.get_scores(y_test, test_preds, labels)

    return test_probs, test_preds, scores


def extend_train_set(X_train, y_train, train_coords,
                     X_test, test_preds, test_coords,
                     test_probs, confidence):

    confident_idxs = np.where(test_probs > confidence)[0]
    X_confident = X_test[confident_idxs, :]
    y_confident = test_preds[confident_idxs, :]
    confident_coords = test_coords[confident_idxs, :]

    check = {}
    for crd in train_coords:
        k = f"{crd[0]}-{crd[1]}-{crd[2]}"
        check[k] = True

    X_train_new = []
    y_train_new = []
    train_coords_new = []

    for i, crd in enumerate(confident_coords):
        k = f"{crd[0]}-{crd[1]}-{crd[2]}"
        if k not in check:
            check[k] = True
            X_train_new.append(X_confident[i, :])
            y_train_new.append(y_confident[i, :])
            train_coords_new.append(confident_coords[i, :])

    X_train_new = np.array(X_train_new)
    y_train_new = np.array(y_train_new)
    train_coords_new = np.array(train_coords_new)

    X_train = np.concatenate((X_train, X_train_new), axis=0)
    y_train = np.concatenate((y_train, y_train_new), axis=0)
    train_coords = np.concatenate((train_coords, train_coords_new), axis=0)

    print(f"X_train: {X_train.shape}")

    return X_train, y_train, train_coords
