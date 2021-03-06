import sys
from time import time
import torch
from os.path import join
from pathlib import Path
import wandb
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.utils import shuffle

from deepmri import Datasets, utils, exp_utils  # noqa: E402
from models.MultiScaleSegmentation import Model  # noqa: E402

def main():
    script_start = time()
    
    parser = argparse.ArgumentParser(
        description='This script can be used to reproduce the results based on features from a 2D CNN with sparse supervised loss.',
        add_help=False)

    parser.add_argument('indir',
                        help='Folder containing all required input files')
    args = parser.parse_args()
    exp_dir = args.indir
    
    # ----------------------------------------------Settings----------------------------------------------
    start = time()
    print("Settings".center(100, "-"))

    cfg = {
        # experiment settings
        "seed": 0,
        "deterministic": True,
        "subj_id": "784565",
        "train": True,
        "inference": True,
        "generate_features": True,
        "run_clf": True,

        # training settings
        "start_epoch": 0,
        "num_epochs": 50,
        "checkpoint": 50,
        "inference_epoch": 50,
        "batch_size": 1,
        "lr": 0.001,
        "weight_decay": 0,
        "labels": ["Other", "CG", "CST", "FX", "CC"],
        "train_slices": [("sagittal", 72), ("coronal", 87), ("axial", 72)],
        # "labels": ["Other", "IFO_left", "IFO_right", "ILF_left", "ILF_right", "SLF_left", "SLF_right"],
        # "train_slices": [("sagittal", 44), ("sagittal", 102), ("coronal", 65)],

        # model settings
        "model_name": "MS2Seg-set2",
        "latent_dim": 44,
        "width": 145,
        "pad": 70,

        # RF classifier settings
        "min_samples_leaf": 8
    }
    wandb.init(project="vcbm2020", config=cfg)

    device_num = 0
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")  # device
    if cfg["deterministic"]:
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = not cfg["deterministic"]  # set False whenever input size varies
    torch.backends.cudnn.deterministic = cfg["deterministic"]

    data_dir = join(exp_dir, "data", cfg["subj_id"], "Diffusion")
    data_path = join(data_dir, "data.nii.gz")
    brain_mask_path = join(data_dir, "nodif_brain_mask.nii.gz")
    tract_masks_path = join(data_dir, "tract_masks/tract_masks.nii.gz")
    #tract_masks_path = join(data_dir, "tract_masks/tract_masks_2.nii.gz")

    print(f"Section time: {time() - start:.2f} seconds.")
    # --------------------------------------------Training set--------------------------------------------
    start = time()

    print("Training set".center(100, "-"))

    trainset = Datasets.OrientationSlicesDataset(data_path=data_path,
                                                 active_mask_path=brain_mask_path,
                                                 tract_masks_path=tract_masks_path,
                                                 width=cfg["width"],
                                                 pad=cfg["pad"],
                                                 labels=cfg["labels"],
                                                 slices=cfg["train_slices"],
                                                 ae=False,
                                                 unsqueeze=True)

    train_cweights = trainset.class_weights.to(device)
    print(f"Train set class weights: {train_cweights}")
    print(f"Total training examples: {len(trainset)}, Batch size: {cfg['batch_size']}")

    print(f"Section time: {time() - start:.2f} seconds.")
    # ----------------------------------------------Test set----------------------------------------------
    start = time()

    print("Test set".center(100, "-"))

    testset = Datasets.OrientationSlicesDataset(data_path=data_path,
                                                active_mask_path=brain_mask_path,
                                                tract_masks_path=tract_masks_path,
                                                width=cfg["width"],
                                                pad=cfg["pad"],
                                                labels=cfg["labels"],
                                                slices="full",
                                                ae=False)
    test_cweights = testset.class_weights.to(device)
    print(f"Test set class weights: {test_cweights}")
    print(f"Total test examples: {len(testset)}")

    print(f"Section time: {time() - start:.2f} seconds.")
    # -------------------------------------------Model settings-------------------------------------------
    start = time()
    print("Model: {}".format(cfg["model_name"]).center(100, "-"))

    model = Model(num_classes=len(cfg["labels"]), width=cfg["width"], pad=cfg["pad"])
    model.to(device)

    if cfg["start_epoch"] != 0:
        model_path = "{}/saved_models/{}_epoch_{}".format(exp_dir, cfg["model_name"], cfg["start_epoch"])
        model.load_state_dict(torch.load(model_path, map_location=f"cuda:{device_num}"))
        print("Loaded pretrained weights starting from epoch {}".format(cfg["start_epoch"]))

    p1 = utils.count_model_parameters(model)
    print(f"Total parameters: {p1[0]}, trainable parameters: {p1[1]}")
    wandb.watch(model)

    print(f"Section time: {time() - start:.2f} seconds.")
    # ----------------------------------------------Training----------------------------------------------
    if cfg["train"]:
        start = time()

        print(f"Training: {cfg['model_name']}".center(100, "-"))

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                               patience=10, verbose=True, min_lr=1e-6)

        for epoch in range(1, cfg["num_epochs"] + 1):

            epoch_start = time()
            model.train()
            epoch_loss = 0.0

            trainloader = shuffle(trainset)

            for batch in trainloader:
                optimizer.zero_grad()

                data_in = batch["data_in"].to(device)
                targets = batch["data_out"].to(device)
                sparsity_weight = batch["sparsity_weight"][:, None, :, :].to(device)

                _, logits = model(data_in)

                loss = F.binary_cross_entropy_with_logits(logits, targets,
                                                          weight=sparsity_weight,
                                                          pos_weight=train_cweights[None, :, None, None],
                                                          reduction="sum")
                n = sparsity_weight.sum() * targets.shape[1]
                loss /= n

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2)
                optimizer.step()

                epoch_loss = epoch_loss + loss.item()

            wandb.log({
                "epoch_time": time() - epoch_start
            })

            if (epoch + cfg["start_epoch"]) % cfg["checkpoint"] == 0:
                models_dir=join(exp_dir, "saved_models")
                Path(models_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           f"{models_dir}/{cfg['model_name']}_epoch_{epoch + cfg['start_epoch']}")

            epoch_loss /= len(trainloader)

            print(f"\nEpoch #{epoch + cfg['start_epoch']}/{cfg['num_epochs'] + cfg['start_epoch']}, "
                  f"\nepoch loss: {epoch_loss:.8f}, "
                  f"\nepoch time: {time() - epoch_start:.2f} seconds")

            # little trick for debugging learning
            ps = utils.model_param_sum(model)

            wandb.log({
                "param_sum": ps,
                "Epoch loss": epoch_loss,
            })

            if scheduler is not None:
                scheduler.step(epoch_loss)
                # scheduler.step(epoch)

        print(f"Section time: {time() - start:.2f} seconds.")
    # ---------------------------------------------Inference----------------------------------------------
    if cfg["inference"]:
        start = time()
        print("Inference".center(100, "-"))
        if cfg["inference_epoch"] != 0:
            model_path = "{}/saved_models/{}_epoch_{}".format(exp_dir, cfg["model_name"], cfg["inference_epoch"])
            model.load_state_dict(torch.load(model_path, map_location=f"cuda:{device_num}"))
            print("Loaded pretrained weights starting from epoch {}".format(cfg["inference_epoch"]))

        print("Full volume: ")
        test_scores = utils.evaluate_conv_classifier(model, testset, device, cfg["labels"], conv_type=2)
        print(test_scores)

        wandb.log(test_scores)
        print(f"Section time: {time() - start:.2f} seconds.")
    # --------------------------------------Generating feature maps---------------------------------------
    if cfg["generate_features"]:
        start = time()
        print("Generating feature maps".center(100, "-"))

        exp_utils.generate_features(exp_dir=exp_dir, subj_id=cfg["subj_id"],
                                    model=model, model_name=cfg["model_name"],
                                    inference_epoch=cfg["inference_epoch"], latent_dim=cfg["latent_dim"],
                                    device=device, dataset=testset, conv_type=2)

        print(f"Section time: {time() - start:.2f} seconds.")
    # -------------------------------------------Run classifier-------------------------------------------
    if cfg["run_clf"]:
        start = time()

        scores = exp_utils.run_clf(min_samples_leaf=cfg["min_samples_leaf"], labels=cfg["labels"],
                                   data_dir=data_dir, train_slices=cfg["train_slices"],
                                   model_name=cfg["model_name"], inference_epoch=cfg["inference_epoch"],
                                   logger=wandb, tract_masks_path=tract_masks_path)
        print(scores)
        wandb.log({
            "RF_scores": scores
        })

        print(f"Section time: {time() - start:.2f} seconds.")
    # ----------------------------------------------------------------------------------------------------
    print(f"Total script time: {time() - script_start:.2f} seconds.")

if __name__ == "__main__":
    main()
