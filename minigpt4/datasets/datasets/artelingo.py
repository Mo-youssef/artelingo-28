import os
from PIL import Image
from collections import OrderedDict
from minigpt4.datasets.datasets.base_dataset import BaseDataset
import pandas as pd
import itertools


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

class Artelingo(BaseDataset, __DisplMixin):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        # """
        # vis_root (string): Root directory of images (e.g. coco/images/)
        # ann_root (string): directory to store the annotation file
        # """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        # self.annotation = []
        for ann_path in ann_paths:
            # self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
            self.annotation = pd.read_csv(ann_path)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        # self.img_ids = {}
        # n = 0
        # for ann in self.annotation:
        #     img_id = ann["image_id"]
        #     if img_id not in self.img_ids.keys():
        #         self.img_ids[img_id] = n
        #         n += 1
        img_ids = self.annotation['image_id'].unique()
        self.img_ids = {img_id: i for i, img_id in enumerate(img_ids)}

    def _add_instance_ids(self, key="instance_id"):
        self.annotation[key] = self.annotation.index
        # for idx, ann in enumerate(self.annotation):
        #     ann[key] = str(idx)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]

        img_file = ann["image_id"]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["utterance"]

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "emotion": ann["emotion"],
            "language": ann["language"],
        }
    
class ArtelingoPair(Artelingo):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        # Create a dictionary mapping image_ids to annotations by language
        self.language_map = {}
        for _, ann in self.annotation.iterrows():
            img_id = ann["image_id"]
            language = ann["language"]
            if img_id not in self.language_map:
                self.language_map[img_id] = {}
            self.language_map[img_id][language] = ann
            
        # Create a list to store combinations of language annotations for each image
        self.combinations_list = []
        for img_id, ann_languages in self.language_map.items():
            combinations = list(itertools.combinations(ann_languages.keys(), 2))
            for lang1, lang2 in combinations:
                self.combinations_list.append({
                    "img_id": img_id,
                    "lang1": lang1,
                    "lang2": lang2,
                    "ann_languages": ann_languages
                })

    def __getitem__(self, index):
        # Extract the combination information based on the index
        combination_info = self.combinations_list[index]
        img_id = combination_info["img_id"]
        lang1 = combination_info["lang1"]
        lang2 = combination_info["lang2"]
        ann_languages = combination_info["ann_languages"]

        # Load and process the image
        image_path = os.path.join(self.vis_root, img_id)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        caption = f'{lang1}: {ann_languages[lang1]["utterance"]}. {lang2}: {ann_languages[lang2]["utterance"]}'

        return {
            "image": image,
            "text_input": caption,
            "language1": lang1,
            "language2": lang2,
            "image_id": self.img_ids[img_id]
        }
        
    def __len__(self):
        # Return the total number of combinations
        return len(self.combinations_list)
    
class ArtelingoMulti(BaseDataset, __DisplMixin):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, percent=1.0, language=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        for ann_path in ann_paths:
            print('Dataset Percent:', percent)
            self.annotation = pd.read_csv(ann_path).sample(frac=percent, random_state=42)
            if language is not None:
                print('Filtering language:', language)
                self.annotation = self.annotation[self.annotation['language'] == language]
            print('Dataset Size:', len(self.annotation))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        img_ids = self.annotation['image_id'].unique()
        self.img_ids = {img_id: i for i, img_id in enumerate(img_ids)}

    def _add_instance_ids(self, key="instance_id"):
        self.annotation[key] = self.annotation.index
        
    def __getitem__(self, index):

        ann = self.annotation.iloc[index]

        img_file = ann["image_name"]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["caption"]

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "emotion": ann["emotion"],
            "language": ann["language"],
        }