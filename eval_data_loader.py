import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset


def read_image_ids_file(path):
    image_ids = []
    seen = set()
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                image_id = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid COCO image ID at {path}:{line_number}: {value!r}"
                ) from exc
            if image_id in seen:
                raise ValueError(
                    f"Duplicate COCO image ID at {path}:{line_number}: {image_id}"
                )
            seen.add(image_id)
            image_ids.append(image_id)

    if not image_ids:
        raise ValueError(f"No COCO image IDs found in {path}")
    return image_ids


def coco_image_id(img_file):
    return int(os.path.splitext(img_file)[0][-12:])


class COCODataSet(Dataset):
    def __init__(self, data_path, trans, image_ids=None):
        self.data_path = data_path
        self.trans = trans

        img_files = os.listdir(self.data_path)
        if image_ids is None:
            random.shuffle(img_files)
        else:
            id_to_file = {
                coco_image_id(img_file): img_file
                for img_file in img_files
                if img_file.lower().endswith(".jpg")
            }
            missing = [
                image_id for image_id in image_ids
                if image_id not in id_to_file
            ]
            if missing:
                preview = ", ".join(str(image_id) for image_id in missing[:10])
                suffix = " ..." if len(missing) > 10 else ""
                raise FileNotFoundError(
                    f"{len(missing)} requested COCO images are missing from "
                    f"{self.data_path}: {preview}{suffix}"
                )
            img_files = [id_to_file[image_id] for image_id in image_ids]
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_id = coco_image_id(img_file)

        image = Image.open(os.path.join(self.data_path, img_file)).convert("RGB")
        image = self.trans(image)

        return {"img_id": img_id, "image": image}


class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans

        image_list, query_list, label_list = [], [], []


        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label}
