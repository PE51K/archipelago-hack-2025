from ultralytics.utils.loss import v8DetectionLoss, FocalLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
import torch.nn.functional as F
import torch


class CustomFocalLoss(FocalLoss):
    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss


class Customv8DetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bce = CustomFocalLoss(alpha=0.25, gamma=2.0)


class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        return Customv8DetectionLoss(self)  # Use our custom loss instead of v8DetectionLoss


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CustomDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model


# Training Configuration
trainer = CustomTrainer(
    overrides=dict(
        resume=True,
        model="solutions/grisha/yolo11_sliced/model_configs/yolo11s-p2.yaml",
        data="data/merged_sliced/sampled_data.yaml",
        project="solutions/grisha/yolo11_sliced/finetuned",
        name="8_512_custom_loss_sliced_dataset_slice_size_1536_focal_loss_yolo11s",
        epochs=15,
        imgsz=512,
        batch=8,
        cls=3.0,
        optimizer="AdamW",
        weight_decay=0.01,
        momentum=0.95,
        lr0=1e-4,
        lrf=2e-1,
        warmup_bias_lr=1e-1,
        warmup_epochs=1,
        cos_lr=True,
        plots=True,
        amp=False,
        save_period=1,
    )
)

# Start training with custom BalancedLoss
trainer.train()
