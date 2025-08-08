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
        self.bce = CustomFocalLoss(alpha=0.3, gamma=1.2)


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
        model="yolo11s.pt",
        data="data/merged_sliced/sampled_data.yaml",
        project="solutions/grisha/yolo11_sliced/finetuned",
        name="8_512_custom_loss_sliced_dataset_slice_size_1536_focal_loss_yolo11s",
        resume=True,
        epochs=30,
        imgsz=512,
        batch=8,
        cls=1.5,
        optimizer="AdamW",
        lr0=1e-4,
        cos_lr=True,
        dropout=0.1,
        plots=True,
        amp=False,
    )
)

# Start training with custom BalancedLoss
trainer.train()
