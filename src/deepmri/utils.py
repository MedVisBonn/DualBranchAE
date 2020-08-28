import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from deepmri.CustomLosses import MaskedLoss
from deepmri import visutils
from matplotlib.lines import Line2D
import sklearn.metrics
import torch.nn.functional as F

sys.path.append("/home/agajan/DVR-Multi-Shell/src/")
from deepmri import dsutils  # noqa: E402


def calc_conv_dim(w, k, s, p):
    """Calculates output dimensions of convolution operator.

    Args:
      w: width
      k: kernel size
      s: stride
      p: padding

    Returns:
        None
    """

    dim = ((w + 2 * p - k) / s) + 1
    print("Conv dim: ", dim, math.floor(dim))


def calc_transpose_conv_dim(w, k, s, p, out_p):
    """Calculates output dimensions of transpose convolution operator.

    Args:
      w: width
      k: kernel
      s: strid
      p: padding
      out_p: out padding

    Returns:
        None
    """

    dim = (w - 1) * s - 2 * p + k + out_p
    print("Deconv dim: ", dim, math.floor(dim))


def count_model_parameters(model):
    """Counts total parameters of model.

    Args:
      model: Pytorch model

    Returns:
        The number of total and trainable parameters.
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def compute_receptive_field(kernel_sizes, stride_sizes):
    """Computes the receptive field of single-path fully convolutional network.
    Calculations are done only for one dimension.
    https://distill.pub/2019/computing-receptive-fields/

    Args:
        kernel_sizes (list of int): List of kernel sizes.
        stride_sizes (list of int): List of stride sizes.

    Returns:
        r0 (int): The size of the receptive field in input needed
            to generate one output feated.
    """

    num_layers = len(kernel_sizes)
    kernel_sizes = [1] + kernel_sizes
    stride_sizes = [1] + stride_sizes

    r0 = 1
    for l in range(1, num_layers + 1):

        p = 1

        for i in range(1, l):
            p *= stride_sizes[i]

        r0 = r0 + p * (kernel_sizes[l] - 1)
    return r0


def compute_receptive_field_borders(feature_idx, kernel_sizes, stride_sizes, padding_sizes):
    """Computes the receptive field fot the given feature index int the output.
    https://distill.pub/2019/computing-receptive-fields/

    Args:
        feature_idx (int): Feature index int the output feature map.
        kernel_sizes (list of int): List of kernel sizes.
        stride_sizes (list of int): List of stride sizes.
        padding_sizes (list of int): List of left padding sizes.

    Returns:
        u_0 (int): Left-most border index.
        v_0 (int): Right-most border index.
    """

    u_L = v_L = feature_idx
    num_layers = len(kernel_sizes)

    kernel_sizes = [1] + kernel_sizes
    stride_sizes = [1] + stride_sizes
    padding_sizes = [1] + padding_sizes

    p_outer = 1
    s_u = 0
    s_v = 0

    for l in range(1, num_layers + 1):
        p_outer *= stride_sizes[l]

        p_inner = 1
        for i in range(1, l):
            p_inner *= stride_sizes[i]

        s_u += (padding_sizes[l] * p_inner)
        s_v += ((1 + padding_sizes[l] - kernel_sizes[l]) * p_inner)

    u_0 = u_L * p_outer - s_u
    v_0 = v_L * p_outer - s_v

    return u_0, v_0


def img_stats(sample, t=None):
    mask = sample['mask']
    x = sample['data']
    y = sample['out']

    # scale_back
    x_back = x * sample['stds'] + sample['means']
    y_back = y * sample['stds'] + sample['means']

    # clamp negative and over maximum values
    y_back = y_back.clamp(min=x_back.min(), max=x_back.max())

    # zero out background
    x[:, mask == 0] = 0
    y[:, mask == 0] = 0
    x_back[:, mask == 0] = 0
    y_back[:, mask == 0] = 0

    masked_mse = MaskedLoss()
    mse = nn.MSELoss(reduction='mean')

    loss = masked_mse(x.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
    scaled_loss = masked_mse(x_back.unsqueeze(0),
                             y_back.unsqueeze(0),
                             mask.unsqueeze(0).unsqueeze(0))

    print("Avg. loss: {:.5f}, Avg. scaled loss: {:.5f}".format(loss.item(), scaled_loss.item()))

    if t is not None:
        slc_x = x_back[t]
        slc_y = y_back[t]
        roi_x = slc_x[mask == 1]
        roi_y = slc_y[mask == 1]
        roi_loss = mse(roi_x, roi_y)
        min_x, max_x = slc_x.min().item(), slc_x.max().item()
        min_y, max_y = slc_y.min().item(), slc_y.max().item()

        visutils.show_slices(
            [slc_x.numpy(), slc_y.numpy(), mask.numpy()],
            titles=['x, min: {:.2f}, max: {:.2f}'.format(min_x, max_x),
                    'y, min: {:.2f}, max: {:.2f}'.format(min_y, max_y),
                    'mask'],
            suptitle='Loss in ROI: {:.2f}'.format(roi_loss.item()),
            cmap='gray'
        )


def dataset_performance(dataset,
                        encoder,
                        decoder,
                        criterion,
                        device,
                        t=0,
                        every_iter=10 ** 10,
                        eval_mode=True,
                        plot=False,
                        masked_loss=False):
    """Calculates average loss on whole dataset.

    Args:
      dataset: Dataset
      encoder: Encoder model.
      decoder: Decoder model.
      criterion: Criterion.
      device: device.
      t: time index for plotting.
      every_iter:  Print statistics every iteration. (Default value = 10 ** 10)
      eval_mode:  Boolean for the model mode. (Default value = True)
      plot:  Boolean to plot. (Default value = False)
      masked_loss: If True loss will be calculated over masked region only.

    Returns:
        None
    """

    if eval_mode:
        encoder.eval()
        decoder.eval()
    else:
        encoder.train()
        decoder.train()

    print("Encoder training mode: {}".format(encoder.training))
    print("Decoder training mode: {}".format(encoder.training))

    start = time.time()

    total_examples = len(dataset)

    min_loss = 10 ** 9
    max_loss = 0

    best = None
    worst = None

    print("Total examples: {}".format(total_examples))

    with torch.no_grad():
        running_loss = 0
        eval_start = time.time()
        for i, data in enumerate(dataset, 1):
            x = data['data'].unsqueeze(0).to(device)
            feature = encoder(x)
            out = decoder(feature)
            mask = data['mask'].unsqueeze(0).unsqueeze(0).to(device)

            if masked_loss:
                loss = criterion(x, out, mask)
            else:
                loss = criterion(x, out)

            if loss.item() < min_loss:
                min_loss = loss.item()
                best = data
                best['out'] = out.detach().cpu().squeeze()
                best['feature'] = feature.detach().cpu().squeeze()

            if loss.item() > max_loss:
                max_loss = loss.item()
                worst = data
                worst['out'] = out.detach().cpu().squeeze()
                worst['feature'] = feature.detach().cpu().squeeze()

            running_loss = running_loss + loss.item()

            if i % every_iter == 0:
                print("Evaluated {}/{} examples. Evaluation time: {:.5f} secs.".format(i,
                                                                                       total_examples,
                                                                                       time.time() - eval_start))
                eval_start = time.time()

        avg_loss = running_loss / total_examples

    print("Evaluated {}/{}, Total evaluation time: {:.5f} secs.".format(total_examples,
                                                                        total_examples,
                                                                        time.time() - start))
    print("Min loss: {:.5f}\nMax loss: {:.5f}\nAvg loss: {:.5f}\nBest Reconstruction: {}\nWorst Reconstruction: {}"
          .format(min_loss,
                  max_loss,
                  avg_loss,
                  best['file_name'],
                  worst['file_name']))

    return best, worst


def evaluate_ae(encoder,
                decoder,
                criterion,
                device,
                trainloader,
                print_iter=False,
                masked_loss=False,
                denoising=False,
                vae=False,
                ):
    """Evaluates AE.

    Args:
      encoder: Encoder model.
      decoder: Decoder model.
      criterion: Criterion.
      device: Device
      trainloader: Train loader
      print_iter:  Print every iteration. (Default value = False)
      masked_loss: If True loss will be calculated over masked region only.
      denoising: If True, denoising AE will be evaluated.
      vae: If True, evaluates VAE.

    Returns:
        Average loss.
    """

    start = time.time()
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        total_examples = 0
        running_loss = 0.0

        for i, batch in enumerate(trainloader, 1):

            # forward
            x = batch['data'].to(device)
            if "out" in batch.keys():
                out = batch["out"]
            else:
                out = x
            out = out.to(device)

            if vae:
                mu, logvar, z = encoder(x)
                y = decoder(z)
                loss = criterion(y, out, mu, logvar)
            else:
                if denoising:
                    h = encoder(batch['noisy_data'].to(device))
                else:
                    h = encoder(x)
                y = decoder(h)

                # calculate loss
                if masked_loss:
                    mask = batch['mask'].unsqueeze(1).to(device)
                    loss = criterion(y, out, mask)
                else:
                    loss = criterion(y, out)

            # track loss
            running_loss = running_loss + loss.item() / batch["data"].size(0)
            total_examples += batch["data"].size(0)

            if print_iter:
                print("Batch #{}, Batch loss: {:.5f}".format(i, loss.item()))
        avg_loss = running_loss / len(trainloader)

    print("Evaluated {} examples, Avg. loss: {:.8f}, Time: {:.5f}".format(total_examples, avg_loss,
                                                                          time.time() - start))
    return avg_loss


def train_ae(encoder,
             decoder,
             criterion,
             optimizer,
             device,
             trainloader,
             valloader,
             num_epochs,
             model_name,
             experiment_dir,
             start_epoch=0,
             vae=False,
             scheduler=None,
             checkpoint=1,
             print_iter=False,
             eval_epoch=5,
             masked_loss=False,
             sparsity=None,
             denoising=False,
             prec=5,
             logger=None
             ):
    """Trains AutoEncoder.

    Args:
      encoder: Encoder model.
      decoder: Decoder model.
      criterion: Criterion.
      optimizer: Optimizer.
      device: Device
      trainloader: Trainloader.
      valloader: Validation loader.
      num_epochs: Number of epochs to train.
      model_name: Model name for saving.
      experiment_dir: Experiment directory with data.
      start_epoch:  Starting epoch. Useful for resuming.(Default value = 0)
      vae: If True, trains Variational Autoencoder.
      scheduler:  Learning rate scheduler.(Default value = None)
      checkpoint:  Save every checkpoint epoch. (Default value = 1)
      print_iter:  Print every iteration. (Default value = False)
      eval_epoch:  Evaluate every eval_epoch epoch. (Default value = 5)
      masked_loss: If True loss will be calculated over masked region only.
      sparsity: If not None, sparsity penalty will be applied to hidden activations.
      denoising: If True, Denoising AE will be trained.
      prec: Error precision.
      logger: Logger.

    Returns:
        None
    """
    print("Training started for {} epochs.".format(num_epochs))
    if sparsity is not None:
        print("Sparsity is on with lambda={}".format(sparsity))
    if denoising:
        print("Training denoising autencoder.")

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        epoch_start = time.time()
        total_examples = 0
        running_loss = 0.0

        iters = 1

        for batch in trainloader:
            iter_time = time.time()

            # forward
            x = batch['data'].to(device)
            if "out" in batch.keys():
                out = batch["out"]
            else:
                out = x
            out = out.to(device)

            if vae:
                mu, logvar, z = encoder(x)
                y = decoder(z)
                loss = criterion(y, out, mu, logvar)
            else:
                if denoising:
                    h = encoder(batch['noisy_data'].to(device))
                else:
                    h = encoder(x)

                y = decoder(h)

                if masked_loss:
                    mask = batch['mask'].unsqueeze(1).to(device)
                    loss = criterion(y, out, mask)
                else:
                    loss = criterion(y, out)

                if sparsity is not None:
                    loss = loss + sparsity * torch.abs(h).sum()

            # zero gradients
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update params
            optimizer.step()

            # track loss
            running_loss = running_loss + loss.item() / batch['data'].size(0)
            total_examples += batch['data'].size(0)
            if print_iter:
                if logger is not None:
                    logger.log({
                        "iter_loss": loss.item()
                    })
                print("Iteration #{}, loss: {:.{}f}, iter time: {}".format(iters,
                                                                           loss.item(),
                                                                           prec,
                                                                           time.time() - iter_time))
            iters += 1

        if (epoch + start_epoch) % checkpoint == 0:
            torch.save(encoder.state_dict(), "{}saved_models/{}_encoder_epoch_{}".format(experiment_dir,
                                                                                         model_name,
                                                                                         epoch + start_epoch))
            torch.save(decoder.state_dict(), "{}saved_models/{}_decoder_epoch_{}".format(experiment_dir,
                                                                                         model_name,
                                                                                         epoch + start_epoch))

        epoch_loss = running_loss / len(trainloader)
        print("Epoch #{}/{},  epoch loss: {:.{}f}, epoch time: {:.5f} seconds".format(epoch + start_epoch,
                                                                                      num_epochs + start_epoch,
                                                                                      epoch_loss,
                                                                                      prec,
                                                                                      time.time() - epoch_start))
        if logger is not None:
            logger.log({
                "epoch": epoch + epoch_start,
                "epoch_loss": epoch_loss
            })
        # evaluate on trainloader
        if epoch % eval_epoch == 0:
            evaluate_ae(encoder, decoder, criterion, device, valloader, print_iter=print_iter,
                        masked_loss=masked_loss, denoising=denoising, vae=vae)

        if scheduler is not None:
            # print("Lr before: {:.8f}".format(optimizer.param_groups[0]['lr']))
            scheduler.step()
            # print("Lr before: {:.8f}".format(optimizer.param_groups[0]['lr']))


def plot_grad_flow(named_parameters):
    """
    Credits https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def evaluate_classifier(model, dataset, device, class_labels, ae=False, batch_size=2**15):

    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    ground_truths = []
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["data"].to(device)
            labels = batch["label"].detach().numpy()

            if ae:
                encoded = model.encode(inputs)
                logits = model.classify(encoded)
            else:
                logits = model.classify(inputs)

            preds = torch.nn.Sigmoid()(logits).cpu().detach().numpy()
            preds[preds > 0.5] = 1
            preds[preds <= 0.5] = 0

            ground_truths.extend(list(labels))
            predictions.extend(list(preds))

    # remove other class
    ground_truths = np.array(ground_truths)[:, 1:]
    predictions = np.array(predictions)[:, 1:]

    test_f1_macro = sklearn.metrics.f1_score(predictions, ground_truths, average="macro")
    test_f1s = sklearn.metrics.f1_score(predictions, ground_truths, average=None)

    for c, f1 in enumerate(test_f1s):
        print("F1 for {}: {:.2f}".format(class_labels[c], f1))
    print("F1_macro: {:.2f}".format(test_f1_macro))


def evaluate_conv_classifier(model, dataset, device, labels, conv_type):

    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    true_masks = np.zeros((145, 174, 145, len(labels)))
    pred_masks = np.zeros((145, 174, 145, len(labels)))

    # everything is initially is labeled as background
    # this is epecially useful for empty slices
    # since they are not passed through network
    true_masks[:, :, :, 0] = 1
    pred_masks[:, :, :, 0] = 1

    with torch.no_grad():
        for batch in dataloader:
            data_in = batch["data_in"].to(device)
            _, logits = model(data_in)
            preds = torch.nn.Sigmoid()(logits).cpu().squeeze(0).numpy()
            mask = batch["data_out"].cpu().squeeze(0).numpy()

            preds[preds > 0.5] = 1
            preds[preds <= 0.5] = 0

            if conv_type == 2:
                coronal_idx = batch["slice_idx"]
                true_masks[:, coronal_idx, :, :] = mask.transpose((1, 2, 0))
                pred_masks[:, coronal_idx, :, :] = preds.transpose((1, 2, 0))
            elif conv_type == 3:
                x, y, z = batch["coords"]
                true_masks[x[0]:x[1], y[0]:y[1], z[0]:z[1], :] = mask.transpose((1, 2, 3, 0))
                pred_masks[x[0]:x[1], y[0]:y[1], z[0]:z[1], :] = preds.transpose((1, 2, 3, 0))

    y_ground_truths, y_predictions = [], []

    # we do not predict for background class
    voxel_coords = np.nonzero(true_masks)[:3]  # take only spatial dims
    voxel_coords = list(zip(*voxel_coords))  # make triples
    voxel_coords = list(set(voxel_coords))  # remove duplicates
    for pt in voxel_coords:
        y_true = list(true_masks[pt[0], pt[1], pt[2], :].astype(np.uint8))
        y_pred = list(pred_masks[pt[0], pt[1], pt[2], :].astype(np.uint8))

        y_ground_truths.append(y_true)
        y_predictions.append(y_pred)

    scores = dsutils.get_scores(y_ground_truths, y_predictions, labels)
    return scores


def model_param_sum(m):
    s = 0
    with torch.no_grad():
        for p in m.parameters():
            s += p.sum().item()
    print(f"Model params. sum: {s}")
    return s


def eval_seg_loss(model, device, dataset, batch_size, cweights):
    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        epoch_loss = 0.0

        for batch in dataloader:
            data_in = batch["data_in"].to(device)
            targets = batch["data_out"].to(device)
            sparsity_weight = batch["sparsity_weight"][:, None, :, :].to(device)

            _, logits = model(data_in)

            loss = F.binary_cross_entropy_with_logits(logits, targets,
                                                      weight=sparsity_weight,
                                                      pos_weight=cweights,
                                                      reduction="sum")
            n = sparsity_weight.sum() * targets.shape[1]
            loss /= n

            epoch_loss = epoch_loss + loss.item()

        epoch_loss /= len(dataloader)
        return epoch_loss

