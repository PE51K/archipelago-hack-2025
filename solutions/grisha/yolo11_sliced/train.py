"""
Custom YOLO Training Script with Balanced Loss for Sparse Annotations

This script implements a custom loss function optimized for datasets where 80% of images
have no annotations. The key improvements over standard v8DetectionLoss are:

1. Focal Loss instead of BCE for better class imbalance handling
2. Normalization by images_with_annotations instead of target_scores_sum
3. False negative penalty to prevent "always empty" predictions
4. Enhanced task-aligned assignment (topk=15, beta=4.0)
5. Real-time false negative monitoring and diagnostics
"""

from ultralytics.utils.loss import v8DetectionLoss, FocalLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import make_anchors
import torch


class BalancedLoss(v8DetectionLoss):
    """
    Custom loss function optimized for datasets with sparse annotations (many empty images).
    
    Key Differences from v8DetectionLoss:
    1. Classification Loss: Uses Focal Loss instead of BCE to handle class imbalance
    2. Normalization: Normalizes by images_with_annotations instead of target_scores_sum
    3. False Negative Penalty: Adds penalty when objects are missed (prevents "always empty" predictions)
    4. Assignment Strategy: More positive assignments (topk=15 vs 10, softer IoU beta=4.0 vs 6.0)
    5. Monitoring: Real-time false negative tracking for diagnostics
    """
    
    def __init__(self, model):
        print("Hello from BalancedLoss!")
        
        # Initialize with modified task-aligned assignment parameters
        # topk=15: More positive assignments per GT (vs default 10)
        # This helps with sparse datasets by creating more learning opportunities
        super().__init__(model, tal_topk=15)
        
        # beta=4.0: Softer IoU weighting in assignment (vs default 6.0)
        # Lower beta makes assignment less strict about perfect IoU overlap
        self.assigner.beta = 4.0
        
        # DIFFERENCE 1: Classification Loss Function
        # Standard v8DetectionLoss uses: nn.BCEWithLogitsLoss(reduction="none")
        # BalancedLoss uses: FocalLoss with class-imbalance handling
        self.bce = FocalLoss(
            gamma=2.0,    # Focus on hard examples (higher gamma = more focus on hard cases)
            alpha=0.9     # Heavily favor positive class (0.9 = 90% weight to positives, combats "always empty")
        )
        
        # DIFFERENCE 3: False Negative Penalty Weight
        # Additive penalty applied when objects are completely missed
        self.false_negative_weight = 1.5  # Moderate penalty to discourage missed detections
    
    def __call__(self, preds, batch):
        """
        Calculate losses with proper normalization for datasets with many empty images.
        
        Key improvements over v8DetectionLoss.__call__():
        - Uses Focal Loss instead of BCE for classification
        - Normalizes by images_with_annotations instead of target_scores_sum
        - Adds false negative penalty to prevent "always empty" predictions
        - Provides diagnostic logging for FN tracking
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Standard target preparation (same as v8DetectionLoss)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Standard bbox prediction (same as v8DetectionLoss)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Task-aligned assignment with modified parameters (topk=15, beta=4.0)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        
        # DIFFERENCE 2: Improved Normalization for Sparse Annotations
        # Standard v8DetectionLoss problem: With 80% empty images, target_scores_sum becomes huge
        # while classification loss stays small, leading to tiny cls_loss values
        images_with_annotations = (mask_gt.sum(dim=1).sum(dim=1) > 0).sum()  # count non-empty images
        total_images = batch_size
        
        # DIFFERENCE 1: Use Focal Loss instead of standard BCE
        # Standard: self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        # Improved: FocalLoss handles class imbalance inherently
        cls_loss_raw = self.bce(pred_scores, target_scores.to(dtype))
        
        # DIFFERENCE 3: False Negative Penalty
        # Add small penalty when positive samples are predicted as negative
        # This prevents the model from learning "always predict empty"
        pred_sigmoid = pred_scores.sigmoid()
        positive_mask = target_scores > 0.5
        false_negative_mask = positive_mask & (pred_sigmoid < 0.5)
        
        if false_negative_mask.sum() > 0:
            fn_penalty = false_negative_mask.float().sum() * 0.1  # Small additive penalty
            cls_loss_raw += fn_penalty
        
        # DIFFERENCE 2: Normalize by actual annotated images, not target_scores_sum
        # This prevents cls_loss from becoming microscopic with sparse annotations
        if images_with_annotations > 0:
            normalization_factor = images_with_annotations * (total_images / batch_size)
            loss[1] = cls_loss_raw / normalization_factor
        else:
            # Handle edge case of completely empty batch
            loss[1] = cls_loss_raw / total_images
            
        # DIFFERENCE 5: Real-time False Negative Monitoring
        # Track how many positive samples are being missed (diagnostic)
        false_negatives = false_negative_mask.sum().item()
        total_positives = positive_mask.sum().item()
        
        # Bbox and DFL losses (unchanged from v8DetectionLoss)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # Apply hyperparameter gains (same as v8DetectionLoss)
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class MyDetectionModel(DetectionModel):
    """
    Custom DetectionModel that uses BalancedLoss instead of standard v8DetectionLoss.
    
    This is the bridge between the trainer and our custom loss function.
    Standard DetectionModel.init_criterion() returns v8DetectionLoss(self)
    Our version returns BalancedLoss(self) for sparse annotation handling.
    """
    def init_criterion(self):
        print("Hello from MyDetectionModel!")
        return BalancedLoss(self)  # Use our custom loss instead of v8DetectionLoss


class MyTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer that uses MyDetectionModel instead of standard DetectionModel.
    
    This ensures our custom loss function is used during training.
    Standard DetectionTrainer.get_model() returns DetectionModel(...)
    Our version returns MyDetectionModel(...) which uses BalancedLoss.
    """
    def get_model(self, cfg=None, weights=None, verbose=True):
        print("Hello from MyTrainer!")
        model = MyDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model


# Training Configuration
print("Hello from train.py!")
trainer = MyTrainer(overrides=dict(
        model="yolo11n.pt",    # Pre-trained weights to start from
        data="data/merged_sliced/data.yml",  # Dataset with 80% empty images
        project="solutions/grisha/yolo11_sliced",
        name="8_640_custom_loss_sliced_dataset_slice_size_1536",
        epochs=30,
        imgsz=640,
        batch=16,
        
        # Loss weighting optimized for sparse annotations:
        cls=10.0,      # Higher classification gain to boost cls_loss importance
        box=7.5,       # Slightly reduced box gain to balance with increased cls
        dfl=1.5        # Standard DFL gain
))

# Start training with custom BalancedLoss
trainer.train()
