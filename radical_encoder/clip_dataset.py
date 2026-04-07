import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

"Adjust according to actual needs"
"A simple dataset example, modified based on the actual dataset"
class RadicalDataset(Dataset):
    def __init__(self, root_dir, csv_path, transform=None, augment=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_path)

        self.transform = transform
        self.augment = augment

        self.structure_keywords = [
            '⿰', '⿱', '⿲', '⿳', '⿴', '⿵', '⿶', '⿷', '⿸', '⿹', '⿺', '⿻'
        ]

    def __len__(self):
        return len(self.data)

    def is_chinese_char(self, text):
        if len(text) != 1:
            return False
        return '\u4e00' <= text <= '\u9fff'

    def is_structure(self, text):
        if text in self.structure_keywords:
            return True
        return False

    def build_prompt(self, text):
        if self.is_structure(text):
            prompt = f"A photo of structure '{text}', commonly seen in characters such as exp1…exp5."
        
        elif self.is_chinese_char(text):
            prompt = f"A photo of radical '{text}', commonly seen in characters such as exp1…exp5."


        return prompt

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.root_dir, row[0])
        character = str(row[1])

        image = Image.open(img_path).convert("RGB")

        if self.augment:
            image = self.augment(image)

        if self.transform:
            image = self.transform(image)

        text_prompt = self.build_prompt(character)

        label = idx

        return image, label, text_prompt