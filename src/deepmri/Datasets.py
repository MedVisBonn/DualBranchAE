import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

from deepmri import dsutils


class DMRIVoxels(Dataset):
    def __init__(self, dmri, mask, subset=None):
        # get coordinates within mask
        self.coords = np.transpose(np.nonzero(mask)).tolist()
        self.coords = [
            crd for crd in self.coords if
            len(np.nonzero(dmri[crd[0], crd[1], crd[2], :])[0]) != 0
        ]

        # take only subset for rapid experimentation
        if subset is not None:
            self.coords = self.coords[:subset]

        self.data = [dmri[crd[0], crd[1], crd[2], :].copy() for crd in self.coords]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        signal = self.data[idx]
        signal = torch.tensor(signal).float()
        coord = self.coords[idx]
        sample = {"data": signal, "coord": coord, "scale_factor": 1}

        return sample


class DMRIVoxelsWithLabels(Dataset):
    def __init__(self, dmri, tracts_mask):
        # get coordinates within mask
        self.coords = np.nonzero(tracts_mask)[:3]  # take only spatial dims
        self.coords = list(zip(*self.coords))  # make triples
        self.coords = list(set(self.coords))  # remove duplicates

        self.data = []
        self.labels = []
        for crd in self.coords:
            x = dmri[crd[0], crd[1], crd[2], :]
            y = tracts_mask[crd[0], crd[1], crd[2], :]

            self.data.append(x)
            self.labels.append(y)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        signal = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).float()
        coord = self.coords[idx]

        sample = {"data": signal, "label": label, "coord": coord}

        return sample


class OrientationSlicesDataset(Dataset):
    def __init__(self, data_path, active_mask_path, tract_masks_path, width, pad, labels, slices,
                 ae=False, unsqueeze=False):

        # load data
        data_in = nib.load(data_path).get_fdata()
        self.scale_factor = data_in.max()
        data_in = data_in / self.scale_factor
        active_mask = nib.load(active_mask_path).get_fdata()

        self.width = width
        self.pad = pad

        self.slice_idxs = []
        self.slices_in = []
        self.slices_out = []
        self.sparsity_weights = []
        self.unsqueeze = unsqueeze

        if ae:
            for idx in range(data_in.shape[1]):
                if active_mask[:, idx, :].sum() > 0:
                    self.slice_idxs.append(idx)
                    slc = data_in[:, idx, :, :].transpose((2, 0, 1))
                    slc_in = np.pad(slc, ((0, 0), (pad, pad), (pad, pad)))
                    self.slices_in.append(torch.tensor(slc_in).float())

                    slc_out = torch.tensor(slc).float()
                    self.slices_out.append(slc_out)

                    self.sparsity_weights.append(torch.tensor(active_mask[:, idx, :]).float())
        else:
            tract_masks = nib.load(tract_masks_path).get_fdata()
            tract_masks = tract_masks[:, :, :, 1:]  # remove background class

            if slices == "full":
                # training set
                segmentation_mask = tract_masks
                sparsity_mask = active_mask
            else:
                # test set
                # make segmentation mask
                segmentation_mask, sparsity_mask = dsutils.make_segmentation_masks(active_mask, tract_masks, slices)

            for idx in range(data_in.shape[1]):
                if sparsity_mask[:, idx, :].sum() > 0:
                    self.slice_idxs.append(idx)
                    slc = data_in[:, idx, :, :].transpose((2, 0, 1))
                    slc_in = np.pad(slc, ((0, 0), (pad, pad), (pad, pad)))
                    self.slices_in.append(torch.tensor(slc_in).float())

                    slc_out = torch.tensor(segmentation_mask[:, idx, :, :].transpose((2, 0, 1))).float()
                    self.slices_out.append(slc_out)

                    self.sparsity_weights.append(torch.tensor(sparsity_mask[:, idx, :]).float())

            self.class_weights = dsutils.get_class_weights(segmentation_mask, labels)
            self.class_weights = torch.tensor(self.class_weights).float()

    def __len__(self):
        return len(self.slice_idxs)

    def __getitem__(self, idx):

        slice_in = self.slices_in[idx]
        slice_out = self.slices_out[idx]
        sparsity = self.sparsity_weights[idx]
        slice_idx = self.slice_idxs[idx]

        if self.unsqueeze:
            slice_in = slice_in.unsqueeze(0)
            slice_out = slice_out.unsqueeze(0)
            sparsity = sparsity.unsqueeze(0)

        sample = {
            "data_in": slice_in,
            "data_out": slice_out,
            "sparsity_weight": sparsity,
            "slice_idx": slice_idx,
            "scale_factor": self.scale_factor
        }

        return sample


class SlidingVolumesDataset(Dataset):
    def __init__(self, data_path, active_mask_path, tract_masks_path, width, pad, labels, slices,
                 ae=False, unsqueeze=False):

        # load data
        data_in = nib.load(data_path).get_fdata()
        self.scale_factor = data_in.max()
        data_in = data_in / self.scale_factor
        active_mask = nib.load(active_mask_path).get_fdata()

        self.unsqueeze = unsqueeze

        if ae:
            sparsity_mask = active_mask
            data_out = data_in
        else:
            tract_masks = nib.load(tract_masks_path).get_fdata()
            tract_masks = tract_masks[:, :, :, 1:]  # remove background class
            if slices == "full":
                # test set
                data_out = tract_masks
                sparsity_mask = active_mask
            else:
                # training set
                train_masks, train_sparsity_mask = dsutils.make_segmentation_masks(active_mask=active_mask,
                                                                                   true_masks=tract_masks,
                                                                                   slices=slices)
                data_out = train_masks
                sparsity_mask = train_sparsity_mask

            self.class_weights = dsutils.get_class_weights(data_out, labels)
            self.class_weights = torch.tensor(self.class_weights).float()

        self.volumes_in, self.sparsity_weights, self.coords = dsutils.make_subvolumes(data_in,
                                                                                      sparsity_mask,
                                                                                      width,
                                                                                      pad)
        self.volumes_out, _, _ = dsutils.make_subvolumes(data_out, sparsity_mask, width, 0)

        # convert to tensors
        self.volumes_in = [torch.tensor(self.volumes_in[idx]).float().permute((3, 0, 1, 2))
                           for idx in range(len(self.volumes_in))]
        self.volumes_out = [torch.tensor(self.volumes_out[idx]).float().permute((3, 0, 1, 2))
                            for idx in range(len(self.volumes_out))]
        self.sparsity_weights = [torch.tensor(self.sparsity_weights[idx]).float()
                                 for idx in range(len(self.sparsity_weights))]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        vol_in = self.volumes_in[idx]
        vol_out = self.volumes_out[idx]
        sparsity = self.sparsity_weights[idx]
        coords = self.coords[idx]

        if self.unsqueeze:
            vol_in = vol_in.unsqueeze(0)
            vol_out = vol_out.unsqeueeze(0)
            sparsity = sparsity.unsqueeze(0)

        return {
            "data_in": vol_in,
            "data_out": vol_out,
            "sparsity_weight": sparsity,
            "coords": coords,
            "scale_factor": self.scale_factor
        }
