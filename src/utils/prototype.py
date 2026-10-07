"""Class-prototype construction and cosine distance, for CG-CPCL.

Ported from ALHVR's utils/Generate_Prototype.py. Only the 2D variants are
included - the 3D ones serve the BraTS/V-Net path, which is out of scope
(ACDC only). Implements Eq. 15 (prototype = masked average pooling of
decoder features) and Eq. 16 (cosine similarity between features and
prototypes) of the paper.
"""
import torch
import torch.nn.functional as F


def getFeatures_2D(fts, mask, region=False):
    """Masked average pooling of one class' features.

    Args:
        fts: C x H x W
        mask: H x W binary mask for a single class
    """
    fts = torch.unsqueeze(fts, 0)
    if torch.is_tensor(region):
        mask = torch.unsqueeze(mask * region, 0)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3))
    else:
        mask = torch.unsqueeze(mask, 0)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
    return masked_fts


def getPrototype_2D(fts, mask, region=False):
    """One prototype per class, averaged over the batch.

    Args:
        fts: B x C x H x W
        mask: B x class x H x W one-hot
    Returns:
        list of length num_classes, each 1 x C
    """
    num_classes = mask.shape[1]
    batch_size = mask.shape[0]
    if torch.is_tensor(region):
        features = [[getFeatures_2D(fts[B, ...], mask[B, C, ...], region[B, ...])
                     for B in range(batch_size)] for C in range(num_classes)]
    else:
        features = [[getFeatures_2D(fts[B, ...], mask[B, C, ...])
                     for B in range(batch_size)] for C in range(num_classes)]
    prototypes = [torch.unsqueeze(torch.sum(torch.cat(class_fts), dim=0), 0) / batch_size
                  for class_fts in features]
    return prototypes


def calDist_2D(fts, prototype, scaler=1.):
    """Cosine similarity between every feature vector and one prototype.

    Args:
        fts: N x C x H x W
        prototype: 1 x C
    """
    dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
    return dist
