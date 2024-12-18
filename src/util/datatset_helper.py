from typing import Literal
import torch
import numpy as np

def suggest_split(
    sample_idx: int,
    num_samples: int,
    train_percentage: float,
    test_equals_val: bool = False,
) -> str:
    if sample_idx / num_samples < train_percentage:
        return "train"
    if test_equals_val:
        return "val"
    test_threshold = train_percentage + (1 - train_percentage) / 2
    if sample_idx / num_samples < test_threshold:
        return "val"
    return "test"

# Taken from https://github.com/usr922/FST/blob/main/semi_seg/fst/dataset/augmentation.py
def generate_cutout_masks(img_size, mask_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio
    mask_cutout_area = mask_size[0] * mask_size[1] / ratio

    im_w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    ma_w = im_w * mask_size[1] // img_size[1]
    im_h = np.round(cutout_area / im_w)
    ma_h = np.round(mask_cutout_area / ma_w)

    im_x_start = np.random.randint(0, img_size[1] - im_w + 1)
    ma_x_start = im_x_start * mask_size[1] // img_size[1]
    im_y_start = np.random.randint(0, img_size[0] - im_h + 1)
    ma_y_start = im_y_start * mask_size[0] // img_size[0]

    im_x_end = int(im_x_start + im_w)
    im_y_end = int(im_y_start + im_h)
    ma_x_end = int(ma_x_start + ma_w)
    ma_y_end = int(ma_y_start + ma_h)

    im_mask = torch.ones(img_size)
    im_mask[im_x_end:im_y_end, im_x_start:im_x_end] = 0
    ma_mask = torch.ones(mask_size)
    ma_mask[ma_x_end:ma_y_end, ma_x_start:ma_x_end] = 0
    return im_mask.long(), ma_mask.long()

def translate_tensor(tensor, h, w, shift_h, shift_w):
    translated_tensor = torch.zeros_like(tensor, device=tensor.device)
    src_y_start = max(0, -shift_h)
    src_y_end = h - max(0, shift_h)
    dst_y_start = max(0, shift_h)
    dst_y_end = h - max(0, -shift_h)

    src_x_start = max(0, -shift_w)
    src_x_end = w - max(0, shift_w)
    dst_x_start = max(0, shift_w)
    dst_x_end = w - max(0, -shift_w)
    if tensor.dim() == 3:
        translated_tensor[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = tensor[:, src_y_start:src_y_end, src_x_start:src_x_end]
    else:
        translated_tensor[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = tensor[src_y_start:src_y_end, src_x_start:src_x_end]
    return translated_tensor

# Taken from https://github.com/usr922/FST/blob/main/semi_seg/fst/dataset/augmentation.py
def generate_unsup_data(data, target):
    assert target.shape[1] == 1
    batch_size, _, im_h, im_w = data.shape
    _, _, ma_h, ma_w = target.shape
    device = data.device

    new_data = []
    new_target = []
    for i in range(batch_size):
        mix_mask_image, mix_mask_label = generate_cutout_masks([im_h, im_w], [ma_h, ma_w], ratio=2)
        mix_mask_image = mix_mask_image.to(device)
        mix_mask_label = mix_mask_label.to(device)
            
        aug_image = data[i] * mix_mask_image
        aug_mask = target[i] * mix_mask_label
        
        # Randomly flip images and masks
        if np.random.random() < 0.5:
            aug_image = torch.flip(aug_image, [1])
            aug_mask = torch.flip(aug_mask, [1])
        if np.random.random() < 0.5:
            aug_image = torch.flip(aug_image, [2])
            aug_mask = torch.flip(aug_mask, [2])
        
        # Shift the image and mask randomly in vertical and horizontal directions
        max_shift_ratio = 0.3
        shift_h_ratio = np.random.random() * max_shift_ratio * 2 - max_shift_ratio
        shift_w_ratio = np.random.random() * max_shift_ratio * 2 - max_shift_ratio
        
        shifted_image = torch.zeros_like(aug_image)
        im_shift_w = int(im_w * shift_w_ratio)
        im_shift_h = int(im_h * shift_h_ratio)
        shifted_image = translate_tensor(aug_image, im_h, im_w, im_shift_h, im_shift_w)

        shifted_mask = torch.zeros_like(aug_mask)
        ma_shift_w = int(ma_w * shift_w_ratio)
        ma_shift_h = int(ma_h * shift_h_ratio)
        shifted_mask = translate_tensor(aug_mask, ma_h, ma_w, ma_shift_h, ma_shift_w)

        # Rotate the image randomly
        rotations = np.random.randint(0, 3)
        rotated_image = torch.rot90(shifted_image, rotations, [1, 2])
        rotated_mask = torch.rot90(shifted_mask, rotations, [1, 2])
        

        new_data.append(
            (
                rotated_image
            ).unsqueeze(0)
        )
        new_target.append(
            (
                rotated_mask
            ).unsqueeze(0)
        )

    new_data, new_target = (
        torch.cat(new_data),
        torch.cat(new_target),
    )
    return new_data, new_target
