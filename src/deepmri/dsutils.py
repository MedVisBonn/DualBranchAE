import os
import time
import nibabel as nib
import numpy as np
import torch
from dipy.tracking import utils as utils_trk
from scipy import ndimage
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dipy.reconst.shm as shm
from dipy.core.sphere import Sphere
from sklearn.cluster import KMeans


def correct_brain_mask(data, brain_mask):
    before = brain_mask.sum()
    for s in range(145):
        for c in range(174):
            for a in range(145):
                if data[s, c, a, :].sum() == 0:
                    brain_mask[s, c, a] = 0
    after = brain_mask.sum()
    print(f"Befor cleaning: {before}. After cleaning: {after}")
    return brain_mask


def create_orientation_dataset(dmri_path, mask_path, save_dir, subj_id, orients=(0, 1, 2)):
    orient_names = ["sagittal", "coronal", "axial"]

    print("Loading file: {}".format(dmri_path))
    data = nib.load(dmri_path).get_fdata()  # 4D data: W x H x D x T
    active_mask = nib.load(mask_path).get_data()

    for orient in orients:

        orient_name = orient_names[orient]
        print("Processing {} orientation...".format(orient_name))
        st = time.time()

        orient_dir = os.path.join(save_dir, orient_name)
        if not os.path.exists(orient_dir):
            os.mkdir(orient_dir)

        for idx in range(data.shape[orient]):
            orient_img = None
            slc_mask = None

            if orient == 0:
                orient_img = data[idx, :, :, :]
                slc_mask = active_mask[idx, :, :]
            elif orient == 1:
                orient_img = data[:, idx, :, :]
                slc_mask = active_mask[:, idx, :]
            elif orient == 2:
                orient_img = data[:, :, idx, :]
                slc_mask = active_mask[:, :, idx]

            if np.sum(slc_mask) <= 0:
                print("{} idx={} is empty. Skipping.".format(orient_name, idx))
            else:
                save_path = os.path.join(orient_dir, "data_{}_{}_idx_{:03d}".format(subj_id, orient_name,idx))
                print("Saving: {}".format(save_path))
                save_path = os.path.join(save_dir, save_path)
                orient_img = orient_img.transpose(2, 0, 1)  # channel x width x height)

                np.savez(save_path,
                         data=orient_img,
                         mask=slc_mask.astype('uint8') if slc_mask is not None else None,
                         idx=idx)

        print("Processed in {:.5f} seconds.".format(time.time() - st))


def get_number_of_points(strmlines):
    """Adapted from https://github.com/MIC-DKFZ/TractSeg/issues/39#issuecomment-496181262

    Args:
      strmlines: nibabel streamlines
    Returns:
        Number of point in streamlines.
    """
    count = 0
    for sl in strmlines:
        count += len(sl)
    return count


def remove_small_blobs(img, threshold=1):
    """Adapted from https://github.com/MIC-DKFZ/TractSeg/issues/39#issuecomment-496181262
    Find blobs/clusters of same label. Only keep blobs with more than threshold elements.
    This can be used for postprocessing.

    Args:
      img: Threshold.
      threshold:  (Default value = 1)

    Returns:
        Binary mask.
    """
    # mask, number_of_blobs = ndimage.label(img, structure=np.ones((3, 3, 3)))  #Also considers diagonal elements for
    # determining if a element belongs to a blob -> not so good, because leaves hardly any small blobs we can remove
    mask, number_of_blobs = ndimage.label(img)
    print('Number of blobs before filtering: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob
    print(counts)

    remove = counts <= threshold
    remove_idx = np.nonzero(remove)[0]

    for idx in remove_idx:
        mask[mask == idx] = 0  # set blobs we remove to 0
    mask[mask > 0] = 1  # set everything else to 1

    mask_after, number_of_blobs_after = ndimage.label(mask)
    print('Number of blobs after filtering: ' + str(number_of_blobs_after))

    return mask


def create_tract_mask(trk_file_path, mask_output_path, ref_img_path, hole_closing=0, blob_th=10):
    """Adapted from https://github.com/MIC-DKFZ/TractSeg/issues/39#issuecomment-496181262
    Creates binary mask from streamlines in .trk file.

    Args:
      trk_file_path: Path for the .trk file
      mask_output_path: Path to save the binary mask.
      ref_img_path: Path to the reference image to get affine and shape
      hole_closing: Integer for closing the holes. (Default value = 0)
      blob_th: Threshold for removing small blobs. (Default value = 10)

    Returns:
        None
    """

    ref_img = nib.load(ref_img_path)
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape

    streamlines = nib.streamlines.load(trk_file_path).streamlines

    # Upsample Streamlines  (very important, especially when using DensityMap Threshold. Without upsampling eroded
    # results)
    print("Upsampling...")
    print("Nr of points before upsampling " + str(get_number_of_points(streamlines)))
    max_seq_len = abs(ref_affine[0, 0] / 4)
    print("max_seq_len: {}".format(max_seq_len))
    streamlines = list(utils_trk.subsegment(streamlines, max_seq_len))
    print("Nr of points after upsampling " + str(get_number_of_points(streamlines)))

    # Remember: Does not count if a fibers has no node inside of a voxel -> upsampling helps, but not perfect
    # Counts the number of unique streamlines that pass through each voxel -> oversampling does not distort result
    dm = utils_trk.density_map(streamlines, ref_affine, ref_shape)

    # Create Binary Map
    dm_binary = dm > 1  # Using higher Threshold problematic, because often very sparse
    dm_binary_c = dm_binary

    # Filter Blobs
    dm_binary_c = remove_small_blobs(dm_binary_c, threshold=blob_th)

    # Closing of Holes (not ideal because tends to remove valid holes, e.g. in MCP)
    if hole_closing > 0:
        size = hole_closing
        dm_binary_c = ndimage.binary_closing(dm_binary_c, structure=np.ones((size, size, size))).astype(dm_binary.dtype)

    # Save Binary Mask
    dm_binary_img = nib.Nifti1Image(dm_binary_c.astype("uint8"), ref_affine)
    nib.save(dm_binary_img, mask_output_path)

    return dm_binary_c.astype("uint8")


def create_multilabel_mask(labels, masks_path, active_mask_path, vol_size=(145, 174, 145)):
    """Creates multilabel binary mask.

    Args:
      labels: List of labels, first element is always 'background', second element is always 'other'.
      masks_path: Path to the binary masks.
      active_mask_path: Path no diffusion brain mask.
      vol_size:  Spatial dimensions of volume. (Default value = (145, 174, 145)

    Returns:
        Multi label binary mask.
    """

    mask_ml = np.zeros((*vol_size, len(labels)))
    background = np.ones(vol_size)  # everything that contains no bundle
    other = nib.load(active_mask_path).get_data().astype('uint8')
    background[other == 1] = 0  # what is within brain is not background ?
    mask_ml[:, :, :, 0] = background

    # first label must always be the 'background'
    for idx, label in enumerate(labels[2:], 2):
        mask = nib.load(os.path.join(masks_path, label + '_binary_mask.nii.gz'))
        mask_data = mask.get_data()  # dtype: uint8
        mask_ml[:, :, :, idx] = mask_data
        other[mask_data == 1] = 0  # remove this from other class

    mask_ml[:, :, :, 1] = other

    return mask_ml.astype('uint8')


def create_data_masks(ml_masks, slice_orients, labels, verbose=True, return_weights=False):
    """Creates multilabel binary mask from full multilabel binary mask for given orienations.
    Args:
        ml_masks: Full multilabel binary mask.
        slice_orients: Slice orientations and their indices to include in mask,
                        e.g., slice_orients = [('sagittal', 72), ('sagittal', 89)]
        labels: Labels.
        verbose: Verbosity.
        return_weights: For class imbalance.
    Returns:
        Multilabel binary mask for given slice orientations.
    """

    data_masks = np.zeros(ml_masks.shape)

    for orient in slice_orients:
        if orient[0] == 'sagittal':
            slc = ml_masks[orient[1], :, :, :]
            data_slc = data_masks[orient[1], :, :, :]
        elif orient[0] == 'coronal':
            slc = ml_masks[:, orient[1], :, :]
            data_slc = data_masks[:, orient[1], :, :]
        elif orient[0] == 'axial':
            slc = ml_masks[:, :, orient[1], :]
            data_slc = data_masks[:, :, orient[1], :]
        else:
            print('Invalid orientation name was given.')
            continue

        for ch, label in enumerate(labels):
            label_mask = slc[:, :, ch]
            data_slc[:, :, ch][np.nonzero(label_mask)] = 1

    total = 0
    weights = []
    for ch, label in enumerate(labels):
        annots = len(np.nonzero(data_masks[:, :, :, ch])[0])
        weights.append(annots)
        total += annots
        if verbose:
            print("\"{}\" has {} annotations.".format(labels[ch], annots))
    if verbose:
        print("Total annotations: {}".format(total))

    if return_weights:
        weights = np.array(weights)
        weights = weights.max() / weights

        return data_masks, weights
    else:
        return data_masks


def create_dataset_from_data_mask(features,
                                  data_masks,
                                  labels=None,
                                  multi_label=False):
    """Creates voxel level dataset.

        Args:
          features: numpy.ndarray of shape WxHxDxK.
          data_masks: numpy.ndarray of shape WxHxDxC: multilabel binary mask volume.
          labels: If not None, names for classes will be used instead of numbers.
          multi_label: If True multi label one hot encoding labels will be generated.

        Returns:
          x_set, y_set, coords_set
        """

    x_set, y_set, coords_set = [], [], []

    if multi_label:
        voxel_coords = np.nonzero(data_masks)[:3]  # take only spatial dims
        voxel_coords = list(zip(*voxel_coords))  # make triples
        voxel_coords = list(set(voxel_coords))  # remove duplicates
        for pt in voxel_coords:
            x = features[pt[0], pt[1], pt[2], :]
            y = data_masks[pt[0], pt[1], pt[2], :]

            x_set.append(x)
            y_set.append(y)
            coords_set.append((pt[0], pt[1], pt[2]))
    else:
        for pt in np.transpose(np.nonzero(data_masks)):
            x = features[pt[0], pt[1], pt[2], :]
            y = pt[3]

            x_set.append(x)
            coords_set.append((pt[0], pt[1], pt[2]))
            if labels is None:
                y_set.append(y)
            else:
                y_set.append(labels[y])

    return np.array(x_set), np.array(y_set), np.array(coords_set)


def get_class_weights(segmentation_masks, labels):
    num_annots = []
    for ch, label in enumerate(labels):
        annots = len(np.nonzero(segmentation_masks[:, :, :, ch])[0])
        print(f"{label}: {annots}")
        num_annots.append(annots)

    num_annots = np.array(num_annots)
    weights = num_annots.max() / num_annots

    return weights


def make_segmentation_masks(active_mask, true_masks, slices):
    """Sampling for coronal slices"""
    segmentation_masks = np.zeros_like(true_masks)
    sparsity_mask = np.zeros((145, 174, 145))

    if slices == "full":
        for s in range(145):
            for c in range(174):
                for a in range(145):
                    if active_mask[s, c, a] == 1:
                        segmentation_masks[s, c, a, :] = true_masks[s, c, a, :].copy()
                        sparsity_mask[s, c, a] = 1
    else:
        for tr_slc in slices:
            if tr_slc[0] == "sagittal":
                s = tr_slc[1]
                for c in range(174):
                    for a in range(145):
                        # we consider voxels only inside brain mask
                        if active_mask[s, c, a] == 1:
                            segmentation_masks[s, c, a, :] = true_masks[s, c, a, :].copy()
                            sparsity_mask[s, c, a] = 1
            elif tr_slc[0] == "coronal":
                c = tr_slc[1]
                for s in range(145):
                    for a in range(145):
                        if active_mask[s, c, a] == 1:
                            segmentation_masks[s, c, a, :] = true_masks[s, c, a, :].copy()
                            sparsity_mask[s, c, a] = 1
            elif tr_slc[0] == "axial":
                a = tr_slc[1]
                for c in range(174):
                    for s in range(145):
                        if active_mask[s, c, a].sum() == 1:
                            segmentation_masks[s, c, a, :] = true_masks[s, c, a, :].copy()
                            sparsity_mask[s, c, a] = 1
            else:
                raise ValueError(f"Wrong orientation: {tr_slc[0]}")

    return segmentation_masks, sparsity_mask


def preds_to_data_mask(preds, voxel_coords, labels, vol_size=(145, 174, 145)):
    """Creates data mask from predictions.

    Args:
        preds: Predictions.
        voxel_coords: Coordinates for each prediction.
        labels: Labels.
        vol_size:  Spatial dimensions of volume. (Default value = (145, 174, 145)

    Return:
        Binary mask.
    """
    data_mask = np.zeros((*vol_size, len(labels)))

    for pred, crd in zip(preds, voxel_coords):
        for ch, v in enumerate(pred):
            if v != 0:
                data_mask[crd[0], crd[1], crd[2], ch] = 1

    return data_mask


def features_from_coords(features, coords, orient, scale=1):
    orient_features = []
    for crd in coords:
        if orient == 'sagittal':
            orient_features.append(features[crd[0], crd[1] // scale, crd[2] // scale, :])
        elif orient == 'coronal':
            orient_features.append(features[crd[1], crd[0] // scale, crd[2] // scale, :])
        elif orient == 'axial':
            orient_features.append(features[crd[2], crd[0] // scale, crd[1] // scale, :])
        else:
            raise ValueError('Unknown orientation.')

    return np.array(orient_features)


def label_stats_from_y(y, labels):
    total = 0
    for idx, label in enumerate(labels):
        labels_count = y[:, idx].sum()
        total += labels_count
        print("Label: {}, has {} annotations.".format(label, labels_count))
    print("Total annotations: {}, labels length: {}, overlapping: {}".format(total, y.shape[0], total - y.shape[0]))


def save_pred_masks(pred_masks, data_dir, subj_id, features_name):
    ref_img_path = os.path.join(data_dir, subj_id, 'nodif_brain_mask.nii.gz')
    ref_img = nib.load(ref_img_path)
    ref_affine = ref_img.affine

    save_path = os.path.join(data_dir, subj_id, 'pred_masks', features_name + "_pred_masks.nii.gz")

    nib.save(nib.Nifti1Image(pred_masks.astype("uint8"), ref_affine), save_path)


def save_one_volume(data_pth, save_pth, vol_idx, binary=True, midx=None):
    img = nib.load(data_pth)
    data = img.get_data()

    one_vol = data[:, :, :, vol_idx]
    if not binary:
        one_vol[one_vol == 1] = midx
    nib.save(nib.Nifti1Image(one_vol, img.affine, img.header), save_pth)


def save_as_one_mask(data_pth, save_pth):
    img = nib.load(data_pth)
    data = img.get_data()
    result = np.zeros((145, 174, 145))

    result[data[:, :, :, 5] == 1] = int(4)  # CC
    result[data[:, :, :, 2] == 1] = int(1)  # CG
    result[data[:, :, :, 3] == 1] = int(2)  # CST
    result[data[:, :, :, 4] == 1] = int(3)  # FX

    nib.save(nib.Nifti1Image(result.astype("uint8"), img.affine, img.header), save_pth)


def save_as_one_mask_for_preds(data_pth, save_pth):
    img = nib.load(data_pth)
    data = img.get_data()
    result = np.zeros((145, 174, 145))

    # result[data[:, :, :, 4] == 1] = int(4)  # CC
    # result[data[:, :, :, 1] == 1] = int(1)  # CG
    result[data[:, :, :, 2] == 1] = int(2)  # CST
    result[data[:, :, :, 3] == 1] = int(3)  # FX

    nib.save(nib.Nifti1Image(result.astype("uint8"), img.affine, img.header), save_pth)


def make_training_slices(seed_slices, it, c, train_slices):
    idx = (-1) ** it * c
    if it % 2 == 0:
        c += 1

    for sl in seed_slices:
        new_slc = (sl[0], sl[1] + idx)
        train_slices.append(new_slc)

    return c, train_slices


def make_pca_volume(data, mask, n_components, pca=None, normalize=True, reconstruct=False, random_state=0):
    print("Making data matrix")
    coords = []
    features = []
    for x in range(145):
        for y in range(174):
            for z in range(145):
                if mask[x, y, z]:
                    coords.append((x, y, z))
                    features.append(data[x, y, z, :])

    if normalize:
        print("Normalizing.")
        features = StandardScaler().fit_transform(features)

    if pca is None:
        print("Performing PCA.")
        pca = PCA(n_components=n_components, random_state=random_state)
        features_reduced = pca.fit_transform(features)
    else:
        print("Pre-computed PCA.")
        features_reduced = pca.transform(features)

    print("Making features volume.")
    features_volume = np.zeros((145, 174, 145, n_components))
    for idx, crd in enumerate(coords):
        features_volume[crd[0], crd[1], crd[2], :] = features_reduced[idx]

    recon_volume = None
    if reconstruct:
        print("Reconstructing")
        recon_volume = np.zeros((145, 174, 145, 288))
        proj = pca.inverse_transform(features_reduced)

        for idx, crd in enumerate(coords):
            recon_volume[crd[0], crd[1], crd[2], :] = proj[idx]

        mse = np.mean((data - recon_volume) ** 2)
        print("MSE: {:.2f}".format(mse))

    return features_volume, pca, recon_volume


def create_voxel_features(model, features_shape, device, dataset, ref_img_pth, save_pth, bs=2**15, vae=False):
    model.eval()
    ref_img = nib.load(ref_img_pth)

    # call with batch size of for given subject
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

    voxel_features = np.zeros(features_shape)

    with torch.no_grad():
        for bid, batch in enumerate(dataloader):
            x = batch['data'].to(device)

            if vae:
                z_mean, z_log_var, encoded, decoded = model(x)
                h = z_mean + torch.exp(z_log_var / 2.)
            else:
                h = model.encode(x)

            for b in range(h.shape[0]):
                crd_0 = batch['coord'][0][b].item()
                crd_1 = batch['coord'][1][b].item()
                crd_2 = batch['coord'][2][b].item()
                voxel_features[crd_0, crd_1, crd_2] = h[b].detach().cpu().squeeze().numpy()
            print(bid, end=" ")
        # np.savez(save_pth, data=voxel_features)
        nib.save(nib.Nifti1Image(voxel_features, ref_img.affine, ref_img.header), save_pth)

    print("Feature maps are saved.")
    return voxel_features


def create_voxel_reconstructions(model, dataset, img_shape, device, ref_img_pth, save_pth, bs=2 ** 15):
    model.eval()
    ref_img = nib.load(ref_img_pth)

    # call with batch size of for given subject
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=10)

    recons = np.zeros(img_shape)

    with torch.no_grad():
        for bid, batch in enumerate(dataloader):
            x = batch['data'].to(device)
            y = model(x)[-1]

            for b in range(y.shape[0]):
                crd_0 = batch['coord'][0][b].item()
                crd_1 = batch['coord'][1][b].item()
                crd_2 = batch['coord'][2][b].item()
                scale_factor = batch['scale_factor'][b].item()
                out_vec = y[b].detach().cpu().numpy()
                recons[crd_0, crd_1, crd_2] = out_vec * scale_factor
            print(bid, end=" ")

        nib.save(nib.Nifti1Image(recons, ref_img.affine, ref_img.header), save_pth)

    print("Reconstructions are saved.")
    return recons


def create_conv_features(model, device, dataset, latent_dim, orient, ref_img_pth, save_pth):
    model.eval()

    print("Processing {} features".format(orient))

    ref_img = nib.load(ref_img_pth)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    orient_shapes = {
        "sagittal": (145, 174, 145, latent_dim),
        "coronal": (174, 145, 145, latent_dim),
        "axial": (145, 145, 174, latent_dim)
    }
    features_map = np.zeros(orient_shapes[orient])

    with torch.no_grad():

        for j, data in enumerate(dataloader):
            slice_idx = int(data["slice_idx"])
            encoded = model.encode(data["data_in"].to(device)).detach().cpu().squeeze(0).numpy()
            features_map[slice_idx] = encoded.transpose(1, 2, 0)

        # transpose features
        if orient == "coronal":
            features_map = features_map.transpose((1, 0, 2, 3))

        elif orient == "axial":
            features_map = features_map.transpose((1, 2, 0, 3))

        # np.savez(save_pth, data=features_map)
        nib.save(nib.Nifti1Image(features_map, ref_img.affine, ref_img.header), save_pth)
    print("Feature maps are saved.")
    return features_map


def create_conv_Q(features, active_mask, n_clusters, ref_img_pth, save_pth):
    ref_img = nib.load(ref_img_pth)

    X = []
    coords = []

    for c in range(174):
        for s in range(145):
            for a in range(145):
                if active_mask[s, c, a] == 1:
                    X.append(features[s, c, a, :])
                    coords.append((s, c, a))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    print("Clustering")
    clusters = kmeans.labels_
    for i in range(n_clusters):
        print(f"Cluster-{i}: {clusters[clusters == i].shape[0]} samples.")

    Q = np.zeros((145, 174, 145, n_clusters))

    for x, crd, cl in zip(X, coords, clusters):
        s = crd[0]
        c = crd[1]
        a = crd[2]
        lbl = [0] * n_clusters
        lbl[cl] = 1
        Q[s, c, a, :] = lbl
    nib.save(nib.Nifti1Image(Q, ref_img.affine), save_pth)
    return Q


def create_conv_reconstructions(model, device, dataset, img_shape, orient, ref_img_pth, save_pth, bs=1):
    model.eval()

    ref_img = nib.load(ref_img_pth)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=10)

    orient_shapes = {
        "sagittal": (img_shape[0], img_shape[1], img_shape[2], img_shape[3]),
        "coronal": (img_shape[1], img_shape[0], img_shape[2], img_shape[3]),
        "axial": (img_shape[2], img_shape[0], img_shape[1], img_shape[3])
    }
    recons = np.zeros(orient_shapes[orient])

    with torch.no_grad():

        for j, data in enumerate(dataloader):
            slice_idx = int(data["slice_idx"])

            _, decoded = model(data["data_in"].to(device))
            decoded = decoded.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            recons[slice_idx] = decoded * data["scale_factor"].item()

        # transpose features
        if orient == "coronal":
            recons = recons.transpose((1, 0, 2, 3))
        elif orient == "axial":
            recons = recons.transpose((1, 2, 0, 3))

    nib.save(nib.Nifti1Image(recons, ref_img.affine, ref_img.header), save_pth)
    print("Reconstructions are saved.")
    return recons


def create_features_from_subvolumes(model, device, latent_dim, dataset, ref_img_pth, save_pth):
    model.eval()

    ref_img = nib.load(ref_img_pth)

    # call with batch size of for given subject
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)
    features_volume = np.zeros((145, 174, 145, latent_dim))

    with torch.no_grad():

        for j, data in enumerate(dataloader):
            vol_in = data["data_in"].to(device)
            coords = data["coords"]
            x0, x1 = coords[0]
            y0, y1 = coords[1]
            z0, z1 = coords[2]
            encoded = model.encode(vol_in).detach().cpu().squeeze(0).numpy()
            features_volume[x0:x1, y0:y1, z0:z1, :] = encoded.transpose(1, 2, 3, 0)

    nib.save(nib.Nifti1Image(features_volume, ref_img.affine, ref_img.header), save_pth)
    print(f"Feature maps are saved. Shape: {features_volume.shape}")
    return features_volume


def create_reconstructions_from_subvolumes(model, device, dataset, ref_img_pth, save_pth, bs=1):
    model.eval()

    ref_img = nib.load(ref_img_pth)

    # call with batch size of for given subject
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=10)

    recons = np.zeros((145, 174, 145, 288))

    with torch.no_grad():

        for j, batch in enumerate(dataloader):
            vol_in = batch["data_in"].to(device)
            _, decoded = model(vol_in)
            coords = batch["coords"]
            scale_factor = batch["scale_factor"].item()
            x0, x1 = coords[0]
            y0, y1 = coords[1]
            z0, z1 = coords[2]
            decoded = decoded.detach().squeeze(0).cpu().numpy()
            recons[x0:x1, y0:y1, z0:z1, :] = scale_factor * decoded.transpose(1, 2, 3, 0)

    nib.save(nib.Nifti1Image(recons, ref_img.affine, ref_img.header), save_pth)
    print("Reconstructions are saved.")
    return recons


def rotate_signal(signal, bvecs, rot_mtrx, sh_order=8, basis_type="descoteaux07"):
    """Rotates raw diffusion signal.

    Args:
        signal (numpy.ndarray): Raw diffusion signal without b0 signal!!!.
        bvecs (numpy.ndarray): Gradient directions.
        rot_mtrx (numpy.ndarray): 3 by 3 rotation matrix.
        sh_order (int): Order of the spherical harmonics. Defaults to 8.
        basis_type (str): Basis type: "descoteaux07" or "tournier07". Defaults to "descoteaux07".

    Returns:
        signal_rot (numpy.ndarray): Rotated signal.
    """
    # 1. Create a Sphere with given gradient directions. Do not consider b0.
    sphere = Sphere(bvecs[:, 0], bvecs[:, 1], bvecs[:, 2])

    # 2. Calculate SH coefficients
    sh_coeffs = shm.sf_to_sh(signal, sphere, sh_order=sh_order, basis_type=basis_type)

    # 3. Rotate gradient directions
    bvecs_rot = np.dot(rot_mtrx, bvecs.T).T

    # 4. Create Sphere with rotated gradient directions
    sphere_rot = Sphere(bvecs_rot[:, 0], bvecs_rot[:, 1], bvecs_rot[:, 2])

    # 5. Calculate rotated signal
    signal_rot = shm.sh_to_sf(sh_coeffs, sphere_rot, sh_order=sh_order, basis_type=basis_type)

    return signal_rot


def make_subvolumes(data, mask, width, pad):
    x_lim, y_lim, z_lim = data.shape[:3]
    data = np.pad(data, ((pad, pad), (pad, pad), (pad, pad), (0, 0)))
    mask = np.pad(mask, ((pad, pad), (pad, pad), (pad, pad)))

    data_volumes = []
    mask_volumes = []
    coords = []

    x = pad
    while x < x_lim + pad:

        # for managing borders
        if (x_lim + pad - x < width) and (x != x_lim + pad):
            x = x_lim + pad - width

        # coords in padded volume
        x_start = x
        x_end = x_start + width

        # actual coords
        coords_x = (x_start - pad, x_end - pad)

        y = pad
        while y < y_lim + pad:
            if (y_lim + pad - y < width) and (y != y_lim + pad):
                y = y_lim + pad - width

            y_start = y
            y_end = y_start + width
            coords_y = (y_start - pad, y_end - pad)

            z = pad
            while z < z_lim + pad:
                if (z_lim + pad - z < width) and (z != z_lim + pad):
                    z = z_lim + pad - width

                z_start = z
                z_end = z_start + width
                coords_z = (z_start - pad, z_end - pad)

                data_vol = data[x_start-pad:x_end+pad, y_start-pad:y_end+pad, z_start-pad:z_end+pad, :]

                # unpadded mask
                mask_vol = mask[x_start:x_end, y_start:y_end, z_start:z_end]

                if mask_vol.sum() > 0:
                    data_volumes.append(data_vol)
                    mask_volumes.append(mask_vol)
                    coords.append((coords_x, coords_y, coords_z))

                z += width
            y += width
        x += width

    return data_volumes, mask_volumes, coords


def get_scores(y_test, test_preds, labels):
    precision, recall, fbeta, support = sklearn.metrics.precision_recall_fscore_support(y_test, test_preds,
                                                                                        average=None)
    scores = {}
    for c in range(len(labels)):
        scores[f"{labels[c]}_precision"] = precision[c]
        scores[f"{labels[c]}_recall"] = recall[c]
        scores[f"{labels[c]}_f1"] = fbeta[c]

    scores["Avg_prec_tracts"] = np.mean(precision[1:])
    scores["Avg_recall_tracts"] = np.mean(recall[1:])
    scores["Avg_f1_tracts"] = np.mean(fbeta[1:])
    print("*For avg. score background and other class scores are ignored!!!.")

    return scores


def create_cluster_maps(model, device, dataset, n_clusters, orient, ref_img_pth, save_pth):
    model.eval()

    print("Processing {} features".format(orient))

    ref_img = nib.load(ref_img_pth)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    orient_shapes = {
        "sagittal": (145, 174, 145, n_clusters),
        "coronal": (174, 145, 145, n_clusters),
        "axial": (145, 145, 174, n_clusters)
    }
    cluster_map = np.zeros(orient_shapes[orient])

    with torch.no_grad():

        for j, data in enumerate(dataloader):
            slice_idx = int(data["slice_idx"])
            _, cluster_logits, _ = model(data["data_in"].to(device))
            cluster_logits = torch.nn.Softmax(dim=1)(cluster_logits.detach())
            cluster_logits = cluster_logits.cpu().squeeze(0).numpy()
            cluster_map[slice_idx] = cluster_logits.transpose(1, 2, 0)

        # transpose features
        if orient == "coronal":
            cluster_map = cluster_map.transpose((1, 0, 2, 3))

        elif orient == "axial":
            cluster_map = cluster_map.transpose((1, 2, 0, 3))

        # np.savez(save_pth, data=features_map)
        nib.save(nib.Nifti1Image(cluster_map, ref_img.affine, ref_img.header), save_pth)
    print("Feature maps are saved.")
    return cluster_map
