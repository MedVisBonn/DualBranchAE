# DualBranchAE

This repository contains the Python code that was used to train and evaluate the machine learning models in the paper "Interactive Classification of Multi-Shell Diffusion MRI With Features From a Dual-Branch CNN Autoencoder" by Agajan Torayev and Thomas Schultz, published at the EG Workshop on Visual Computing in Biology and Medicine 2020.

The purpose of publishing this code in its current form is to allow other researchers in our field to reproduce our results. At this point, it has not been optimized to serve as an easy-to-use software package.

## Relevant scripts to reproduce results

### Dependencies
The code has been successfully tested with `Python 3.7.6`, `torch 1.6.0`, `wandb 0.9.5`, and `dipy 1.1.1`.

### Multi-scale autoencoders
This is the main approach proposed in our work: Unsupervised representation learning using a dual-branch autoencoder, followed by fast supervised learning with a random forest.

The autoencoder models are in `models/MultiScaleAE.py` and `MultiScaleAE3d.py`.

Results can be reproduced with the scripts `exp_conv_ae.py` and `exp_conv3d_ae.py`.

Each script
<ol>
<li>Trains the 2D or 3D multi-scale architecture.</li>
<li>Generates features for the RF classifier.</li>
<li>Runs RF classifier with given labels and training slices.</li>
<li>Additionally, generates reconstructions and calculates MSE.</li>
</ol>

The configurations for each script can be adjusted through a `cfg` dictionary in the beginning of the script. The scripts expect a single command-line
argument, the path of the folder that contains the required data. In particular, it needs to contain the subfolder `data/<subj_id>/Diffusion`, where `subj_id` should match the setting in `cfg`. That subfolder needs to contain:
<ol>
<li><code>data.nii.gz</code> The 4D dMRI data</li>
<li><code>nodif_brain_mask.nii.gz</code> A 3D binary mask indicating the location of brain tissue</li>
<li><code>tract_masks/tract_masks.nii.gz</code> A 4D binary mask indicating the labels for the random forest. This needs to match the <code>labels</code> in <code>cfg</code>.</li>
</ol>

### Multi-scale segmentation networks
This is an alternative that is evaluated in Section 5.4 of our paper: Instead of using an unsupervised autoencoder loss, we experimented with a sparse supervised loss that accounts for the available annotations. It was still advantageous to apply a random forest to the resulting features.

The segmentation models are in `models/MultiScaleSegmentation.py` and `MultiScaleSegmentation3d.py`. 

They can be run with the scripts `exp_conv_segment.py` and `exp_conv3d_segment.py`.

Each script
<ol>
<li>Trains the 2D or 3D multi-scale architecture.</li>
<li>Makes inference using segmentation network.</li>
<li>Generates features for the RF classifier.</li>
<li>Runs RF classifier with given labels and training slices.</li>
</ol>

### `deepmri` library
`deepmri` library contains codes for the `Datasets`, utility functions for experiments, data, architectures, 
RF classifier and visualization.

### `util_scripts`:
<ul>
<li><code>create_pca_features.py</code> Creates PCA features for dMRI data.</li>
<li><code>pca_k_search.py</code> Creates statistics for different numbers of PCA components, to facilitate selecting the best.</li>
<li><code>create_shore_features.py</code> Creates SHORE features for dMRI data. This is only required for the supplementary results. In addition to the same files required by the other scripts, it requires `bvals` and `bvecs` files corresponding to the `data.nii.gz`.</li>
<li><code>rf_classifier.py</code> Trains and evaluates a RF classifier given pre-computed features (e.g., PCA or SHORE from the scripts above).</li>
</ul>
