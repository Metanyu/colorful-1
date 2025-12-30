from src.colour.soft_encode import soft_encode, soft_encode_fast
import torch.nn.functional as F

def ab_to_z(ab, hull, use_fast=True):
    ab = F.interpolate(ab, scale_factor=0.25, mode='bilinear', align_corners=False)
    ab = ab.permute(0, 2, 3, 1)
    if use_fast:
        return soft_encode_fast(ab, centroids=hull)
    return soft_encode(ab, centroids=hull)