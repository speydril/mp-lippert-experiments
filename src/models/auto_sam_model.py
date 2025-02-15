from dataclasses import dataclass
from typing import Literal, Optional

from sklearn.metrics import roc_auc_score, roc_curve
import torch
from src.models.segment_anything.modeling.sam import SamBatched
from src.args.yaml_config import YamlConfig
from src.util.polyp_transform import get_polyp_transform
from src.models.segment_anything.build_sam import (
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
)
from src.models.base_model import BaseModel, ModelOutput, Loss
from src.datasets.base_dataset import Batch
from pydantic import BaseModel as PDBaseModel
from torch.nn import BCEWithLogitsLoss as BCELoss
from src.models.auto_sam_prompt_encoder.model_single import ModelEmb
from torch.nn import functional as F
import numpy as np

from src.util.image_util import calc_iou, extract_patch, join_patches
from src.util.eval_util import (
    get_dice_ji,
    get_optimal_threshold_auc,
    get_optimal_threshold_iou,
)


class AutoSamModelArgs(PDBaseModel):
    sam_model: Literal["vit_h", "vit_l", "vit_b"]
    sam_checkpoint: str = "/dhc/groups/mp2024cl2/sam_vit_b.pth"
    hard_net_cp: str = "/dhc/groups/mp2024cl2/hardnet68.pth"
    hard_net_arch: int = 68
    depth_wise: bool = False
    Idim: int = 512


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


@dataclass
class SAMBatch(Batch):
    original_size: torch.Tensor
    image_size: torch.Tensor
    metadata: Optional[dict] = None


@dataclass
class SAMSampleFileReference:
    img_path: str
    gt_path: str


# Source of most of this code: https://github.com/talshaharabany/AutoSAM
class AutoSamModel(BaseModel[SAMBatch]):
    def __init__(self, config: AutoSamModelArgs, image_encoder_no_grad=True):
        super().__init__()
        self.sam = sam_model_registry[config.sam_model](
            checkpoint=config.sam_checkpoint
        )
        self.bce_loss = BCELoss()
        self.prompt_encoder = ModelEmb(
            hard_net_arch=config.hard_net_arch,
            depth_wise=config.depth_wise,
            hard_net_cp=config.hard_net_cp,
        )
        self.config = config
        self.image_encoder_no_grad = image_encoder_no_grad

    def forward(self, batch: SAMBatch) -> ModelOutput:
        Idim = self.config.Idim
        orig_imgs_small = F.interpolate(
            batch.input, (Idim, Idim), mode="bilinear", align_corners=True
        )
        dense_embeddings = self.prompt_encoder(orig_imgs_small)
        masks = sam_call(
            batch.input, self.sam, dense_embeddings, self.image_encoder_no_grad
        )

        return ModelOutput(masks)

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
        masks = self.sam.postprocess_masks(
            normalized_logits, input_size=input_size, original_size=original_size
        )
        gts = self.sam.postprocess_masks(
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

        binary_gts = gts.clone()
        binary_gts[binary_gts > 0.5] = 1
        binary_gts[binary_gts <= 0.5] = 0
        roc = roc_auc_score(
            binary_gts.detach().flatten().cpu(), masks.detach().flatten().cpu()
        )
        auc_threshold = get_optimal_threshold_auc(
            binary_gts.flatten().cpu(), masks.detach().flatten().cpu()
        )
        auc_masks = masks.clone()
        auc_masks[auc_masks > auc_threshold] = 1
        auc_masks[auc_masks <= auc_threshold] = 0
        auc_dice_score, auc_IoU = get_dice_ji(
            auc_masks.squeeze().detach().cpu().numpy(),
            gts.squeeze().detach().cpu().numpy(),
        )

        iou_threshold = get_optimal_threshold_iou(
            gts.squeeze().cpu().numpy(), masks.detach().squeeze().cpu().numpy()
        )
        iou_masks = masks.clone()
        iou_masks[iou_masks > iou_threshold] = 1
        iou_masks[iou_masks <= iou_threshold] = 0
        iou_dice_score, iou_IoU = get_dice_ji(
            iou_masks.squeeze().detach().cpu().numpy(),
            gts.squeeze().detach().cpu().numpy(),
        )

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
                "roc_auc": float(roc),
                "optimal_iou_threshold": float(iou_threshold),
                "optimal_iou_threshold_dice_score": iou_dice_score,
                "optimal_iou_threshold_IoU": iou_IoU,
                "optimal_auc_threshold": float(auc_threshold),
                "optimal_auc_threshold_dice_score": auc_dice_score,
                "optimal_auc_threshold_IoU": auc_IoU,
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
        Idim = self.config.Idim

        image_tensor = transform.apply_image_torch(img)
        input_size = tuple(image_tensor.shape[1:3])
        input_images = transform.preprocess(image_tensor).unsqueeze(dim=0).cuda()

        orig_imgs_small = F.interpolate(
            input_images, (Idim, Idim), mode="bilinear", align_corners=True
        )
        dense_embeddings = self.prompt_encoder.forward(orig_imgs_small)
        with torch.no_grad():
            mask = norm_batch(
                sam_call(
                    input_images, self.sam, dense_embeddings, image_encoder_no_grad=True
                )
            )

        mask = self.sam.postprocess_masks(
            mask, input_size=input_size, original_size=original_size
        )
        mask = mask.squeeze().cpu().numpy()
        mask = (255 * mask).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return image, mask

    def segment_image_from_file(
        self, image_path: str, patches: Optional[Literal[4, 16]] = None
    ):
        import cv2

        image = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        yaml_config = YamlConfig().config
        pixel_mean, pixel_std = (
            yaml_config.fundus_pixel_mean,
            yaml_config.fundus_pixel_std,
        )
        if patches is not None:
            img_patches = [extract_patch(image, i) for i in range(4)]

            if patches == 4:
                masks = [
                    self.segment_image(
                        img, pixel_mean, pixel_std, yaml_config.fundus_resize_img_size
                    )[1]
                    for img in img_patches
                ]
            else:
                masks = []
                for patch in img_patches:
                    sub_patches = [extract_patch(patch, i) for i in range(4)]
                    masks.append(
                        join_patches(
                            [
                                self.segment_image(
                                    sub_patch,
                                    pixel_mean,
                                    pixel_std,
                                    yaml_config.fundus_resize_img_size,
                                )[1]
                                for sub_patch in sub_patches
                            ]
                        )
                    )
            mask = join_patches(masks)
            return image, mask

        return self.segment_image(
            image, pixel_mean, pixel_std, yaml_config.fundus_resize_img_size
        )

    def segment_and_write_image_from_file(
        self,
        image_path: str,
        output_path: str,
        mask_opacity: float = 0.4,
        gts_path: Optional[str] = None,
        threshold=0.5,
        patches: Optional[Literal[4, 16]] = None,
    ):
        import cv2
        from PIL import Image

        image, mask = self.segment_image_from_file(image_path, patches)
        if gts_path is not None:
            with Image.open(gts_path) as im:
                gts = np.array(im.convert("RGB"))
        else:
            gts = np.zeros_like(mask)
        mask[mask > 255 * threshold] = 255
        mask[mask <= 255 * threshold] = 0
        overlay = (
            np.array(mask) * np.array([1, 0, 1]) + np.array(gts) * np.array([0, 1, 0])
        ).astype(image.dtype)
        output_image = cv2.addWeighted(
            image, 1 - mask_opacity, overlay, mask_opacity, 0
        )
        if threshold != 0.5:
            cv2.putText(
                output_image,
                f"Threshold: {threshold:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
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


def compute_dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            "image": img,
            "original_size": original_size,
            "image_size": input_size,
            "point_coords": None,
            "point_labels": None,
        }
        batched_input.append(singel_input)
    return batched_input


def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = (
        x.view(bs, -1)
        .min(dim=1)[0]
        .repeat(1, 1, 1, 1)
        .permute(3, 2, 1, 0)
        .repeat(1, 1, Isize, Isize)
    )
    max_value = (
        x.view(bs, -1)
        .max(dim=1)[0]
        .repeat(1, 1, 1, 1)
        .permute(3, 2, 1, 0)
        .repeat(1, 1, Isize, Isize)
    )
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def sam_call(
    batched_input, sam: SamBatched, dense_embeddings, image_encoder_no_grad=True
):
    with torch.set_grad_enabled(not image_encoder_no_grad):
        input_images = batched_input
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks
