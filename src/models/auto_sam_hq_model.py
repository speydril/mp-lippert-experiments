from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel as PDBaseModel
from sklearn.metrics import roc_auc_score, roc_curve

from sympy import im, use
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss as BCELoss
from torch.nn import functional as F
from src.models.auto_sam_model import SAMBatch, get_dice_ji
from src.args.yaml_config import YamlConfig
from src.models.base_model import BaseModel, ModelOutput, Loss
from src.models.auto_sam_prompt_encoder.model_single import ModelEmb
from src.util.polyp_transform import get_polyp_transform
import numpy as np
from src.models.segment_anything_hq.segment_anything_training import sam_model_registry
from src.models.segment_anything_hq.segment_anything_training.modeling.sam import Sam
from src.models.segment_anything_hq.segment_anything_training.modeling.mask_decoder import (
    MaskDecoder,
)
from src.models.segment_anything_hq.segment_anything_training.modeling.transformer import (
    TwoWayTransformer,
)
from src.models.segment_anything_hq.segment_anything_training.modeling.common import (
    LayerNorm2d,
)
from src.util.image_util import calc_iou, extract_patch, join_patches



class AutoSamHQModelArgs(PDBaseModel):
    sam_model: Literal["vit_h", "vit_l", "vit_b"]
    sam_hq_checkpoint: str = "/dhc/groups/mp2024cl2/sam_vit_b.pth"
    hard_net_cp: str = "/dhc/groups/mp2024cl2/hardnet68.pth"
    hard_net_arch: int = 68
    depth_wise: bool = False
    Idim: int = 512
    use_hq_token_only: bool = True


# Source of most of this code: https://github.com/talshaharabany/AutoSamHQ
class AutoSamHQModel(BaseModel[SAMBatch]):
    def __init__(self, config: AutoSamHQModelArgs, image_encoder_no_grad=True):
        super().__init__()
        self.sam = sam_model_registry[config.sam_model](
            checkpoint=config.sam_hq_checkpoint
        )

        self.bce_loss = BCELoss()
        self.prompt_encoder = ModelEmb(
            hard_net_arch=config.hard_net_arch,
            depth_wise=config.depth_wise,
            hard_net_cp=config.hard_net_cp,
        )
        self.config = config
        self.image_encoder_no_grad = image_encoder_no_grad
        self.mask_decoder = MaskDecoderHQ(config.sam_model)

    def forward(self, batch: SAMBatch) -> ModelOutput:
        Idim = self.config.Idim
        orig_imgs_small = F.interpolate(
            batch.input, (Idim, Idim), mode="bilinear", align_corners=True
        )
        dense_embeddings = self.prompt_encoder(orig_imgs_small)
        masks = self.sam_call(
            batch.input,
            self.sam,
            dense_embeddings,
            self.image_encoder_no_grad,
            use_hq_token_only=self.config.use_hq_token_only,
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
        roc = roc_auc_score(binary_gts.flatten().cpu(), masks.flatten().cpu())
        auc_threshold = self.get_optimal_threshold_auc(
            binary_gts.flatten().cpu(), masks.flatten().cpu()
        )
        auc_masks = masks.clone()
        auc_masks[auc_masks > auc_threshold] = 1
        auc_masks[auc_masks <= auc_threshold] = 0
        auc_dice_score, auc_IoU = get_dice_ji(
            auc_masks.squeeze().detach().cpu().numpy(),
            gts.squeeze().detach().cpu().numpy(),
        )

        iou_threshold = self.get_optimal_threshold_iou(
            gts.squeeze().cpu().numpy(), masks.squeeze().cpu().numpy()
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
                "iou_threshold": float(iou_threshold),
                "iou_dice_score": iou_dice_score,
                "iou_IoU": iou_IoU,
                "auc_threshold": float(auc_threshold),
                "auc_dice_score": auc_dice_score,
                "auc_IoU": auc_IoU,
            },
        )

    def get_optimal_threshold_auc(self, y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]

    def get_optimal_threshold_iou(self, y_true, y_score):
        best_iou = 0
        best_threshold = 0
        for threshold in np.linspace(0, 1, 100):
            pred_masks = y_score > threshold
            _, iou = get_dice_ji(pred_masks, y_true)
            if iou > best_iou:
                best_iou = iou
                best_threshold = threshold
        return best_threshold

    def sam_call(
        self,
        batched_input,
        sam: Sam,
        dense_embeddings,
        image_encoder_no_grad=True,
        use_hq_token_only=False,
    ):
        with torch.set_grad_enabled(not image_encoder_no_grad):
            input_images = batched_input
            image_embeddings, interm_embeddings = sam.image_encoder(input_images)

            sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(
                points=None, boxes=None, masks=None
            )
            empty_sparse_embeddings = [
                sparse_embeddings_none for _ in range(len(input_images))
            ]

        image_pes = [
            sam.prompt_encoder.get_dense_pe() for _ in range(len(input_images))
        ]
        masks_hq = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pes,
            sparse_prompt_embeddings=empty_sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=use_hq_token_only,
            interm_embeddings=interm_embeddings,
        )
        return masks_hq

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
            un_normalized_mask = self.sam_call(
                input_images,
                self.sam,
                dense_embeddings,
                image_encoder_no_grad=True,
                use_hq_token_only=True,
            )
            mask = norm_batch(un_normalized_mask)

        mask = self.sam.postprocess_masks(
            mask, input_size=input_size, original_size=original_size
        )
        mask = mask.squeeze().cpu().numpy()
        mask = (255 * mask).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return image, mask

    def segment_image_from_file(self, image_path: str, patches: Optional[Literal[4, 16]] = None):
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

        image, mask = self.segment_image_from_file(image_path, patches=patches)
        if gts_path is not None:
            with Image.open(gts_path) as im:
                gts = np.array(im.convert("RGB"))
        else:
            gts = np.zeros_like(mask)

        mask[mask > (255 * threshold)] = 255
        mask[mask <= (255 * threshold)] = 0
        overlay = (
            np.array(mask) * np.array([1, 0, 1]) + np.array(gts) * np.array([0, 1, 0])
        ).astype(image.dtype)
        output_image = cv2.addWeighted(
            image, 1 - mask_opacity, overlay, mask_opacity, 0
        )
        if threshold is not 0.5:
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
    if isinstance(x, (tuple, list)):
        x = x[0]

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


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            num_multimask_outputs=3,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        assert model_type in ["vit_b", "vit_l", "vit_h"]

        checkpoint_dict = {
            "vit_b": "/dhc/groups/mp2024cl2/sam_vit_b_maskdecoder.pth",
        }
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n, p in self.named_parameters():
            p.requires_grad = False

        transformer_dim = 256
        vit_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 8, kernel_size=2, stride=2
            ),
        )

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
    ):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """

        vit_features = interm_embeddings[0].permute(
            0, 3, 1, 2
        )  # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(
            vit_features
        )

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature=hq_features[i_batch].unsqueeze(0),
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks, 0)
        iou_preds = torch.cat(iou_preds, 0)

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[
                torch.arange(masks_multi.size(0)), max_iou_idx
            ].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]

        masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :]

        if hq_token_only:
            return masks_hq
        else:
            return masks_sam, masks_hq

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight],
            dim=0,
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = (
            self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        )

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(
                    self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                )
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:, :4] @ upscaled_embedding_sam.view(b, c, h * w)).view(
            b, -1, h, w
        )
        masks_ours = (hyper_in[:, 4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(
            b, -1, h, w
        )
        masks = torch.cat([masks_sam, masks_ours], dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
