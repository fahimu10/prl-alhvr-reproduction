"""Per-volume validation metrics, ported from ALHVR's utils/val_2d.py.

Kept faithful to the original, including its `if len(output) > 1` guard:
for a plain UNet the output is a single Tensor whose len() is the batch
size (1 here), so the guard is a no-op, but UNet_fea_aux returns a
4-tuple and the guard selects the primary segmentation head. Keeping it
means this function works unchanged for the ALHVR runs.
"""
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred = pred.copy()
    gt = gt.copy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    return 0, 0


def test_single_volume(image, label, model, classes, device, patch_size=(256, 256)):
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    model.eval()
    for ind in range(image.shape[0]):
        slice_ = image[ind, :, :]
        x, y = slice_.shape
        slice_ = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=0)
        inp = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(inp)
            if len(output) > 1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
