import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from model import K210_YOLO
from dataset_xml import VOCDataset, collate_fn

import argparse


# =====================================
# YOLO LOSS
# =====================================

class YOLOLoss(nn.Module):

    def __init__(self,
                 num_classes=3,
                 lambda_coord=5,
                 lambda_noobj=0.5):

        super().__init__()

        self.num_classes = num_classes

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):

        # pred:
        # [B,H,W,A,5+C]

        obj_mask = target[...,4] > 0
        noobj_mask = ~obj_mask

        # =========================
        # xy
        # =========================

        pred_xy = torch.sigmoid(pred[...,0:2])

        target_xy = target[...,0:2]

        # =========================
        # wh
        # =========================

        pred_wh = torch.clamp(pred[...,2:4], min=1e-6)

        target_wh = torch.clamp(target[...,2:4], min=1e-6)

        # =========================
        # conf
        # =========================

        pred_conf = pred[...,4]

        target_conf = target[...,4]

        # =========================
        # class
        # =========================

        pred_cls = pred[...,5:]

        target_cls = target[...,5:]

        # =========================
        # coord loss
        # =========================

        if obj_mask.sum() > 0:

            xy_loss = self.mse(
                pred_xy[obj_mask],
                target_xy[obj_mask]
            )

            wh_loss = self.mse(
                torch.sqrt(pred_wh[obj_mask]),
                torch.sqrt(target_wh[obj_mask])
            )

            coord_loss = self.lambda_coord * (
                xy_loss + wh_loss
            )

        else:
            coord_loss = torch.tensor(
                0.0,
                device=pred.device
            )

        # =========================
        # conf loss
        # =========================

        conf_obj_loss = self.bce(
            pred_conf[obj_mask],
            target_conf[obj_mask]
        ) if obj_mask.sum() > 0 else 0

        conf_noobj_loss = self.bce(
            pred_conf[noobj_mask],
            target_conf[noobj_mask]
        )

        conf_loss = (
            conf_obj_loss +
            self.lambda_noobj * conf_noobj_loss
        )

        # =========================
        # cls loss
        # =========================

        if obj_mask.sum() > 0:

            cls_loss = self.bce(
                pred_cls[obj_mask],
                target_cls[obj_mask]
            )

        else:
            cls_loss = torch.tensor(
                0.0,
                device=pred.device
            )

        total_loss = (
            coord_loss +
            conf_loss +
            cls_loss
        )

        return total_loss, {
            "coord": float(coord_loss),
            "conf": float(conf_loss),
            "cls": float(cls_loss),
            "total": float(total_loss)
        }


# =====================================
# TRAIN
# =====================================

def train(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)

    # =====================================
    # classes
    # =====================================

    class_names = [
        "tri",
        "sp",
        "cir"
    ]

    # =====================================
    # dataset
    # =====================================

    train_dataset = VOCDataset(
        root_dir=args.data_path,
        class_names=class_names,
        input_size=args.input_size,
        split='train',
        transform=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # =====================================
    # model
    # =====================================

    model = K210_YOLO(
        num_classes=len(class_names),
        num_anchors=5
    ).to(device)

    # =====================================
    # init weights
    # =====================================

    def init_weights(m):

        if isinstance(m, nn.Conv2d):

            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity='relu'
            )

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):

            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    print("weights initialized")

    # =====================================
    # optimizer
    # =====================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    criterion = YOLOLoss(
        num_classes=len(class_names)
    )

    best_loss = 1e9

    # =====================================
    # train loop
    # =====================================

    for epoch in range(args.epochs):

        model.train()

        epoch_loss = 0

        for batch_idx, (imgs, targets, _) in enumerate(train_loader):

            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss, loss_dict = criterion(
                outputs,
                targets
            )

            if torch.isnan(loss):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                10.0
            )

            optimizer.step()

            epoch_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Batch [{batch_idx}] "
                f"Loss: {loss_dict['total']:.4f}"
            )

        avg_loss = epoch_loss / len(train_loader)

        scheduler.step()

        print(
            f"\nEpoch [{epoch+1}] "
            f"Average Loss: {avg_loss:.4f}\n"
        )

        # =====================================
        # save best
        # =====================================

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_dir,
                    "best.pth"
                )
            )

            print("saved best model")


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        default=r'D:\yolov2\dataset'
    )

    parser.add_argument(
        '--input_size',
        type=int,
        default=224
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=150
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='./weights'
    )

    args = parser.parse_args()

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    train(args)