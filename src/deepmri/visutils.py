import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_slices(slices,
                suptitle="Visualization",
                titles=('Saggital', 'Coronal', 'Axial'),
                figsize=(10, 5),
                fontsize=24,
                cmap=None):
    """Function to display row of image slices

    Args:
      slices: Slices to show
      suptitle:  (Default value = "Visualization")
      titles:  (Default value = ('Saggital', 'Coronal', 'Axial'):
      figsize:  (Default value = (10, 5):
      fontsize:  (Default value = 24)
      cmap:  (Default value = None)

    Returns:
        None
    """

    plt.rcParams["figure.figsize"] = figsize
    fig, axes = plt.subplots(1, len(slices))
    fig.suptitle(suptitle, fontsize=fontsize)
    for i, slc in enumerate(slices):
        axes[i].set_title(titles[i])
        axes[i].imshow(slc.T, cmap=cmap, origin="lower", interpolation='none')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    # fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


def show_masked_slices(slices,
                       masks,
                       suptitle="Visualization",
                       titles=('Saggital', 'Coronal', 'Axial'),
                       figsize=(10, 5),
                       fontsize=24,
                       cmap=matplotlib.cm.gray,
                       mask_color='red',
                       alpha=0.9):
    """Function to display row of image slices

    Args:
      slices: Slices to show
      masks: Masks.
      suptitle:  Sup title. (Default value = "Visualization")
      titles:  (Default value = ('Saggital', 'Coronal', 'Axial'):
      figsize:  (Default value = (10, 5):
      fontsize:  (Default value = 24)
      cmap:  (Default value = matplotlib.cm.gray)
      mask_color:  Color for mask. (Default value = 'red')
      alpha:  Opacity. (Default value = 0.9)

    Returns:
        None
    """

    plt.rcParams["figure.figsize"] = figsize
    fig, axes = plt.subplots(1, len(slices))
    fig.suptitle(suptitle, fontsize=fontsize)
    for i, slc in enumerate(slices):
        masked_img = np.ma.array(slc.T, mask=masks[i].T)
        cmap.set_bad(mask_color, alpha=alpha)

        axes[i].set_title(titles[i])
        axes[i].imshow(masked_img, cmap=cmap, origin="lower")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


def show_one_slice(slc,
                   title="One Slice",
                   figsize=(10, 5),
                   fontsize=12,
                   cmap=None):
    """

    Args:
      slc: 
      title:  (Default value = "One Slice")
      figsize:  (Default value = (10)
      5): 
      fontsize:  (Default value = 12)
      cmap:  (Default value = None)

    Returns:

    """
    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(slc.T, cmap=cmap, origin="lower")
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def show_one_masked_slice(img,
                          mask,
                          cmap=matplotlib.cm.gray,
                          title="Masked Image",
                          figsize=(10, 5),
                          fontsize=12,
                          mask_color='red',
                          alpha=0.9):
    """

    Args:
      img: 
      mask: 
      cmap:  (Default value = matplotlib.cm.gray)
      title:  (Default value = "Masked Image")
      figsize:  (Default value = (10)
      5): 
      fontsize:  (Default value = 12)
      mask_color:  (Default value = 'red')
      alpha:  (Default value = 0.9)

    Returns:

    """
    masked_img = np.ma.array(img, mask=mask)
    cmap.set_bad(mask_color, alpha=alpha)
    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(masked_img, origin='lower', cmap=cmap)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def show_tiled_images(images, titles, n_rows, n_cols,  figsize=(36, 16),
                      suptitle='Title', title_x=0.5, title_y=0.9,
                      fontsize=18, zero_space=False, cmap=None):
    """Shows tiled images in grid."""

    fig = plt.figure(figsize=figsize)
    axes = [fig.add_subplot(n_rows, n_cols, i+1) for i in range(len(images))]
    for c, ax in enumerate(axes):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(images[c], origin='lower', cmap=cmap)
        ax.set_title(titles[c])
        ax.axis('off')

    if zero_space:
        fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(suptitle, x=title_x, y=title_y, fontsize=fontsize)
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Credits: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Args:
      cm: 
      classes: 
      normalize:  (Default value = False)
      title:  (Default value = 'Confusion matrix')
      cmap:  (Default value = plt.cm.Blues)

    Returns:

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def visualize_masks(dmri_data,
                    ml_masks_list,
                    labels,
                    x_coord,
                    y_coord,
                    z_coord,
                    t,
                    suptitles,
                    mask_color='red'
                    ):
    """Visualize masks.
    Args:
        dmri_data: Diffusion MRI data
        ml_masks_list: List of masks to visualize.
        labels: Names of labels in masks.
        x_coord: x coordinate.
        y_coord: y coordinate.
        z_coord: z coordinate.
        t: time coordinate.
        suptitles: Suptitles.
        mask_color: Mask color.

    Returns:
        None
    """
    # show binary masks
    slices = [
        dmri_data[x_coord, :, :, t],
        dmri_data[:, y_coord, :, t],
        dmri_data[:, :, z_coord, t]
    ]

    for ch in range(len(labels)):
        for idx, ml_masks in enumerate(ml_masks_list):
            masks = [
                ml_masks[x_coord, :, :, ch],
                ml_masks[:, y_coord, :, ch],
                ml_masks[:, :, z_coord, ch]
            ]

            show_masked_slices(slices,
                               masks,
                               suptitle=suptitles[idx] + labels[ch],
                               mask_color=mask_color)


def visualize_preds(dmri,
                    tract_masks,
                    pred_masks_list,
                    labels,
                    tract_name,
                    orientation,
                    ylabels,
                    slice_idxs=None,
                    n_rows=2,
                    figsize=None,
                    t=1,
                    fontsize=12):
    print("Visualizing {} in {} orientation".format(tract_name, orientation))
    gt_imgs = []  # ground truth
    pr_imgs_list = []  # predictions

    class_idxs = {k: v for v, k in enumerate(labels)}
    tract_idx = class_idxs[tract_name]

    if slice_idxs is None:
        if (orientation == "Sagittal") or (orientation == "Axial"):
            slice_idxs = [54, 71, 72, 73, 90]
        elif orientation == "Coronal":
            slice_idxs = [42, 86, 87, 88, 132]

    # add ground truth images
    for idx in slice_idxs:
        if orientation == "Sagittal":
            gt_img = dmri[idx, :, :, t].copy()
            gt_msk = tract_masks[idx, :, :, tract_idx].copy()

            if figsize is None:
                figsize = (10, 5.15)
        elif orientation == "Coronal":
            gt_img = dmri[:, idx, :, t].copy()
            gt_msk = tract_masks[:, idx, :, tract_idx].copy()

            if figsize is None:
                figsize = (10, 6.15)

        elif orientation == "Axial":
            gt_img = dmri[:, :, idx, t].copy()
            gt_msk = tract_masks[:, :, idx, tract_idx].copy()

            if figsize is None:
                figsize = (10, 7.4)
        else:
            raise ValueError('Unknown orientation.')
        gt_out = np.ma.array(gt_img, mask=gt_msk)
        gt_imgs.append(gt_out)

    # add prediction image
    for pred_masks in pred_masks_list:
        for idx in slice_idxs:
            if orientation == "Sagittal":
                pred_img = dmri[idx, :, :, t].copy()
                pred_msk = pred_masks[idx, :, :, tract_idx].copy()
            elif orientation == "Coronal":
                pred_img = dmri[:, idx, :, t].copy()
                pred_msk = pred_masks[:, idx, :, tract_idx].copy()
            elif orientation == "Axial":
                pred_img = dmri[:, :, idx, t].copy()
                pred_msk = pred_masks[:, :, idx, tract_idx].copy()
            else:
                raise ValueError('Unknown orientation.')

            pred_out = np.ma.array(pred_img, mask=pred_msk)
            pr_imgs_list.append(pred_out)

    titles = ["{}: {}".format(orientation, idx) for idx in slice_idxs] * n_rows

    imgs = gt_imgs + pr_imgs_list

    cmap = matplotlib.cm.gray
    cmap.set_bad("red")

    n_cols = len(slice_idxs)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0, wspace=0)

    y = 0
    for r in range(n_rows):
        for c in range(n_cols):
            j = n_cols * r + c
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(imgs[j].T, origin="lower", cmap=cmap)
            if r == 0:
                ax.set_title(titles[j], fontsize=fontsize)
            if c == 0:
                plt.ylabel(ylabels[y], fontsize=fontsize)
                y += 1

            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()
