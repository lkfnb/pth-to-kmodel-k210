import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


class VOCDataset(Dataset):

    def __init__(self,root_dir,class_names,input_size=224,split='train',transform=True):
        self.root_dir = root_dir
        self.split = split
        self.input_size = input_size
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.transform = transform

        # YOLO参数
        self.num_anchors = 5

        #  grid size
        self.S = 7
        # stride
        self.stride = input_size / self.S
        # k210优化anchors
        self.anchors = torch.tensor([
            [8, 10],
            [14, 18],
            [22, 32],
            [40, 56],
            [72, 96],
        ], dtype=torch.float32)
        # 路径


        self.img_dir = os.path.join(
            root_dir,
            split,
            'img'
        )

        self.xml_dir = os.path.join(
            root_dir,
            split,
            'xml'
        )
        # 获取所有图片
        all_imgs = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(
                (
                    '.jpg',
                    '.jpeg',
                    '.png',
                    '.bmp'
                )
            )
        ])

        self.img_files = []

        for f in all_imgs:

            xml_name = os.path.splitext(f)[0] + '.xml'

            xml_path = os.path.join(
                self.xml_dir,
                xml_name
            )

            if os.path.exists(xml_path):

                self.img_files.append(f)

            else:
                print(
                    f"Warning: XML not found for {f}"
                )

        print(
            f"[{split}] "
            f"Found {len(self.img_files)} samples"
        )

    def __len__(self):
        return len(self.img_files)

    # =====================================
    # parse xml
    # =====================================

    def parse_xml(self, xml_path):

        tree = ET.parse(xml_path)

        root = tree.getroot()

        size = root.find('size')

        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        boxes = []
        labels = []

        for obj in root.findall('object'):

            name = obj.find('name').text

            if name not in self.class_names:
                continue

            label = self.class_names.index(name)

            bbox = obj.find('bndbox')

            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            boxes.append([
                xmin,
                ymin,
                xmax,
                ymax
            ])

            labels.append(label)

        return img_w, img_h, boxes, labels

    # =====================================
    # build target
    # =====================================

    def build_target(self, boxes, labels):

        H = W = self.S

        target = torch.zeros(
            H,
            W,
            self.num_anchors,
            5 + self.num_classes
        )

        for box, label in zip(boxes, labels):

            x1, y1, x2, y2 = box

            w = x2 - x1
            h = y2 - y1

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # =========================
            # grid
            # =========================

            grid_x = int(cx / self.stride)
            grid_y = int(cy / self.stride)

            # 防止越界
            grid_x = min(grid_x, W - 1)
            grid_y = min(grid_y, H - 1)

            if grid_x < 0 or grid_x >= W:
                continue

            if grid_y < 0 or grid_y >= H:
                continue

            # =========================
            # best anchor
            # =========================

            box_wh = torch.tensor([w, h])

            inter = torch.min(
                box_wh,
                self.anchors
            ).prod(dim=1)

            union = (
                box_wh.prod()
                + self.anchors.prod(dim=1)
                - inter
            )

            iou = inter / (union + 1e-6)

            best_anchor = iou.argmax()

            # =========================
            # YOLO target
            # =========================

            target[
                grid_y,
                grid_x,
                best_anchor,
                0
            ] = cx / self.stride - grid_x

            target[
                grid_y,
                grid_x,
                best_anchor,
                1
            ] = cy / self.stride - grid_y

            #  相对anchor比例
            anchor_w = self.anchors[
                best_anchor
            ][0]

            anchor_h = self.anchors[
                best_anchor
            ][1]

            target[
                grid_y,
                grid_x,
                best_anchor,
                2
            ] = w / anchor_w

            target[
                grid_y,
                grid_x,
                best_anchor,
                3
            ] = h / anchor_h

            # conf
            target[
                grid_y,
                grid_x,
                best_anchor,
                4
            ] = 1.0

            # class
            target[
                grid_y,
                grid_x,
                best_anchor,
                5 + label
            ] = 1.0

        return target

    # =====================================
    # get item
    # =====================================

    def __getitem__(self, idx):

        img_name = self.img_files[idx]

        img_path = os.path.join(
            self.img_dir,
            img_name
        )

        xml_name = os.path.splitext(
            img_name
        )[0] + '.xml'

        xml_path = os.path.join(
            self.xml_dir,
            xml_name
        )

        # =========================
        # image
        # =========================

        img = Image.open(
            img_path
        ).convert('RGB')

        orig_w, orig_h = img.size

        # =========================
        # xml
        # =========================

        _, _, boxes, labels = self.parse_xml(
            xml_path
        )

        # =========================
        # resize
        # =========================

        img = img.resize(
            (
                self.input_size,
                self.input_size
            ),
            Image.BILINEAR
        )

        # =========================
        # resize bbox
        # =========================

        scale_x = self.input_size / orig_w
        scale_y = self.input_size / orig_h

        scaled_boxes = []

        for box in boxes:

            x1, y1, x2, y2 = box

            scaled_boxes.append([
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ])

        # =========================
        # image tensor
        # =========================

        img_array = np.array(
            img,
            dtype=np.float32
        ) / 255.0

        img_tensor = torch.from_numpy(
            img_array
        ).permute(2,0,1)

        # =========================
        # target
        # =========================

        target = self.build_target(
            scaled_boxes,
            labels
        )

        return img_tensor, target, img_name


# =====================================
# collate
# =====================================

def collate_fn(batch):

    imgs = torch.stack(
        [item[0] for item in batch],
        dim=0
    )

    targets = torch.stack(
        [item[1] for item in batch],
        dim=0
    )

    img_names = [
        item[2]
        for item in batch
    ]

    return imgs, targets, img_names