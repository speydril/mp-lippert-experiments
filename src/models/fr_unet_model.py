from typing import Callable, Optional, Tuple, cast
import numpy as np
import torch
import torch.nn as nn
from torch import nn
from timm.layers.weight_init import trunc_normal_
from torch.nn import BCEWithLogitsLoss as BCELoss
from src.args.yaml_config import YamlConfig
from src.models.auto_sam_model import (
    SAMBatch,
    compute_dice_loss,
    norm_batch,
)
from src.models.base_model import BaseModel, Loss, ModelOutput
from torch.nn import functional as F

from src.util.polyp_transform import get_polyp_transform
from src.util.image_util import calc_iou
from src.util.eval_util import get_dice_ji


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module: nn.Module):
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.neg_slope)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class conv(nn.Module):
    def __init__(self, in_c, out_c, dp: float = 0):
        super(conv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class feature_fuse(nn.Module):
    def __init__(self, in_c, out_c):
        super(feature_fuse, self).__init__()
        self.conv11 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2
        )
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        out = self.norm(x1 + x2 + x3)
        return out


class up(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_c, out_c, kernel_size=2, padding=0, stride=2, bias=False
            ),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class down(nn.Module):
    def __init__(self, in_c, out_c, dp=0):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class block(nn.Module):
    def __init__(
        self, in_c, out_c, dp: float = 0, is_up=False, is_down=False, fuse=False
    ):
        super(block, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        if fuse == True:
            self.fuse = feature_fuse(in_c, out_c)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = conv(out_c, out_c, dp=dp)
        if self.is_up == True:
            self.up = up(out_c, out_c // 2)
        if self.is_down == True:
            self.down = down(out_c, out_c * 2)

    def forward(self, x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up == False and self.is_down == False:
            return x
        elif self.is_up == True and self.is_down == False:
            x_up = self.up(x)
            return x, x_up
        elif self.is_up == False and self.is_down == True:
            x_down = self.down(x)
            return x, x_down
        else:
            x_up = self.up(x)
            x_down = self.down(x)
            return x, x_up, x_down


class _FR_UNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        num_channels=1,
        feature_scale=2,
        dropout=0.2,
        fuse=True,
        out_ave=True,
    ):
        super(_FR_UNet, self).__init__()
        self.out_ave = out_ave
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.block1_3 = block(
            num_channels, filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse
        )
        self.block1_2 = block(
            filters[0], filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse
        )
        self.block1_1 = block(
            filters[0] * 2, filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse
        )
        self.block10 = block(
            filters[0] * 2, filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse
        )
        self.block11 = block(
            filters[0] * 2, filters[0], dp=dropout, is_up=False, is_down=True, fuse=fuse
        )
        self.block12 = block(
            filters[0] * 2,
            filters[0],
            dp=dropout,
            is_up=False,
            is_down=False,
            fuse=fuse,
        )
        self.block13 = block(
            filters[0] * 2,
            filters[0],
            dp=dropout,
            is_up=False,
            is_down=False,
            fuse=fuse,
        )
        self.block2_2 = block(
            filters[1], filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse
        )
        self.block2_1 = block(
            filters[1] * 2, filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse
        )
        self.block20 = block(
            filters[1] * 3, filters[1], dp=dropout, is_up=True, is_down=True, fuse=fuse
        )
        self.block21 = block(
            filters[1] * 3, filters[1], dp=dropout, is_up=True, is_down=False, fuse=fuse
        )
        self.block22 = block(
            filters[1] * 3, filters[1], dp=dropout, is_up=True, is_down=False, fuse=fuse
        )
        self.block3_1 = block(
            filters[2], filters[2], dp=dropout, is_up=True, is_down=True, fuse=fuse
        )
        self.block30 = block(
            filters[2] * 2, filters[2], dp=dropout, is_up=True, is_down=False, fuse=fuse
        )
        self.block31 = block(
            filters[2] * 3, filters[2], dp=dropout, is_up=True, is_down=False, fuse=fuse
        )
        self.block40 = block(
            filters[3], filters[3], dp=dropout, is_up=True, is_down=False, fuse=fuse
        )
        self.final1 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True
        )
        self.final2 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True
        )
        self.final3 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True
        )
        self.final4 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True
        )
        self.final5 = nn.Conv2d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=True
        )
        self.fuse = nn.Conv2d(5, num_classes, kernel_size=1, padding=0, bias=True)
        self.apply(cast(Callable[[nn.Module], None], InitWeights_He))

    def forward(self, x):
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1)
        )
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(x_down3_1)
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))
        if self.out_ave == True:
            output = (
                self.final1(x1_1)
                + self.final2(x10)
                + self.final3(x11)
                + self.final4(x12)
                + self.final5(x13)
            ) / 5
        else:
            output = self.final5(x13)

        return output


from pydantic import BaseModel as PBaseModel


class FRUnetArgs(PBaseModel):
    dropout: float = 0.2
    Idim: int = 512


class FRUnet(BaseModel[SAMBatch]):
    def __init__(self, config: FRUnetArgs):
        super().__init__()
        self.unet = _FR_UNet(num_channels=3, dropout=config.dropout)
        self.bce_loss = BCELoss()
        self.config = config

    def forward(self, batch: SAMBatch) -> ModelOutput:
        return ModelOutput(logits=self.unet(batch.input))

    def compute_loss(self, outputs: ModelOutput, batch: SAMBatch) -> Loss:
        assert batch.target is not None

        normalized_logits = norm_batch(outputs.logits)
        size = outputs.logits.shape[2:]
        gts_sized = F.interpolate(
            (
                batch.target.unsqueeze(dim=1)
                if batch.target.dim() != outputs.logits.dim()
                else batch.target
            ),
            size,
            mode="nearest",
        )

        bce = self.bce_loss.forward(outputs.logits, gts_sized)
        dice_loss = compute_dice_loss(normalized_logits, gts_sized)
        loss_value = bce + dice_loss

        input_size = tuple(batch.image_size[0][-2:].int().tolist())
        original_size = tuple(batch.original_size[0][-2:].int().tolist())
        gts = batch.target.unsqueeze(dim=0)
        masks = self.postprocess_masks(
            normalized_logits, input_size=input_size, original_size=original_size
        )
        gts = self.postprocess_masks(
            batch.target.unsqueeze(dim=0) if batch.target.dim() != 4 else batch.target,
            input_size=input_size,
            original_size=original_size,
        )
        masks = F.interpolate(
            masks,
            (self.config.Idim, self.config.Idim),
            mode="bilinear",
            align_corners=True,
        )
        gts = F.interpolate(
            gts, (self.config.Idim, self.config.Idim), mode="bilinear"
        )  # was mode=nearest in original code
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0
        dice_score, IoU = get_dice_ji(
            masks.squeeze().detach().cpu().numpy(), gts.squeeze().detach().cpu().numpy()
        )

        return Loss(
            loss_value,
            {
                "dice+bce_loss": loss_value.detach().item(),
                "dice_loss": dice_loss.detach().item(),
                "bce_loss": bce.detach().item(),
                "dice_score": dice_score,
                "IoU": IoU,
            },
        )

    def segment_image(
        self,
        image: np.ndarray,
        pixel_mean: tuple[float, float, float],
        pixel_std: tuple[float, float, float],
        img_encoder_size: int,
    ):
        import cv2
        from .segment_anything.utils.transforms import ResizeLongestSide

        _, test_transform = get_polyp_transform()
        img, _ = test_transform(image, np.zeros_like(image))
        original_size = tuple(img.shape[1:3])

        transform = ResizeLongestSide(img_encoder_size, pixel_mean, pixel_std)

        image_tensor = transform.apply_image_torch(img)
        input_size = tuple(image_tensor.shape[1:3])
        input_images = transform.preprocess(image_tensor).unsqueeze(dim=0).cuda()

        with torch.no_grad():
            mask = norm_batch(self.unet(input_images))

        mask = self.postprocess_masks(
            mask, input_size=input_size, original_size=original_size
        )
        mask = mask.squeeze().cpu().numpy()
        mask = (255 * mask).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return image, mask

    def segment_image_from_file(self, image_path: str):
        import cv2

        image = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        yaml_config = YamlConfig().config
        pixel_mean, pixel_std = (
            yaml_config.fundus_pixel_mean,
            yaml_config.fundus_pixel_std,
        )
        return self.segment_image(
            image, pixel_mean, pixel_std, yaml_config.fundus_resize_img_size
        )

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=True,
        )
        masks = masks[..., : int(input_size[0]), : int(input_size[1])]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=True)
        return masks

    def segment_and_write_image_from_file(
        self,
        image_path: str,
        output_path: str,
        mask_opacity: float = 0.4,
        gts_path: Optional[str] = None,
    ):
        import cv2
        from PIL import Image

        image, mask = self.segment_image_from_file(image_path)
        if gts_path is not None:
            with Image.open(gts_path) as im:
                gts = np.array(im.convert("RGB"))
        else:
            gts = np.zeros_like(mask)
        mask[mask > 255 / 2] = 255
        mask[mask <= 255 / 2] = 0
        overlay = (
            np.array(mask) * np.array([1, 0, 1]) + np.array(gts) * np.array([0, 1, 0])
        ).astype(image.dtype)
        output_image = cv2.addWeighted(
            image, 1 - mask_opacity, overlay, mask_opacity, 0
        )
        iou = calc_iou(mask, gts)
        cv2.putText(
            output_image,
            f"IoU: {iou:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
