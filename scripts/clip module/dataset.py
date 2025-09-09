import torch
from torch.utils.data import Dataset
import ast
import pandas as pd
from augmentation import TextAugmentor

class BlendshapeDataset_all(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.text_column = "Sentence"
        self.emotion_column = "Emotions"
        self.description_column = "Description"
        self.blendshape_column = "blendshapes"
        self.text_augmentor = TextAugmentor()  # Initialize augmentor once

    def __len__(self):
        return len(self.dataframe)

    def text_augmentation(self, text):
        """Return a list of augmented versions of the text"""
        augmented_texts = self.text_augmentor.augment(text)
        return [text] + augmented_texts  # Include the original text too

    def blendshape_augmentor(self, blendshape):
        pertubation_vec = torch.rand(51) * 0.15  # Perturbation of 0.0 to 0.2
        return torch.clamp(blendshape + pertubation_vec, max=1.0)

    def __getitem__(self, idx):
        # Get text-related data
        t = str(self.dataframe.iloc[idx][self.text_column] or '')
        e = str(self.dataframe.iloc[idx][self.emotion_column] or '')
        d = str(self.dataframe.iloc[idx][self.description_column] or '')

        # Generate all augmented samples
        text_options_aug = self.text_augmentation(t) + \
                       self.text_augmentation(e) + \
                       self.text_augmentation(d)
        
        # Generate all normal samples
        #text_options = [t, e, d]

        # Process blendshapes
        blendshapes_str = self.dataframe.iloc[idx][self.blendshape_column]
        blendshapes = ast.literal_eval(blendshapes_str)
        blendshapes_tensor = torch.tensor(blendshapes[1:], dtype=torch.float32)
        blendshape_augmented = self.blendshape_augmentor(blendshapes_tensor)
        #blendshape_normal = blendshapes_tensor

        return text_options_aug, blendshape_augmented