import os
from collections import OrderedDict
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionInstructDataset, CaptionEvalDataset
from PIL import Image

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class RXRCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None # image does not exist
        image = self.vis_processor(image)
        # TODO: pensar cambiar "caption" por "description"
        caption = self.text_processor(ann["caption"])
        return {
            "image": image,
            "text_input": caption,
            "image_id": ann["image_id"]
        }

class RXRCaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }

# TODO: ver si me sirve esto realmente
class RXRCaptionInstructDataset(CaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data