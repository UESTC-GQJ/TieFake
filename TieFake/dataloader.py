import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, utils
from transformers import BertTokenizer
import ssl
import sys
from emotion.extract_emotion import cut_words_from_text,extract_publisher_emotion
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class FakedditImageDataset(Dataset):
    """The Fakeddit image dataset class"""

    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_frame = pd.read_csv(csv_file, encoding='ansi', delimiter='\t')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.csv_frame.loc[idx,'id']!=np.NaN:
            img_name = str(int(self.csv_frame.loc[idx, 'id'])) + '.jpg'
            img_path = os.path.join(self.root_dir, img_name)
        else:
            return
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            label = self.csv_frame.loc[idx, 'label']
            if self.transform:
                image = self.transform(image)
            return (image, label)
        except Exception:
            print(f"Corrupted image {img_name}")

class FakedditEmoDataset(Dataset):
    """The text + emo dataset class"""

    def __init__(self, csv_file):
        self.csv_frame = pd.read_csv(csv_file, encoding='ansi', delimiter='\t')
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def __len__(self):
        return len(self.csv_frame)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            sent = self.csv_frame.loc[idx, 'clean_text']
            t=cut_words_from_text(sent)
            emo=extract_publisher_emotion(sent,t)
            bert_encoded_dict = self.bert_tokenizer.encode_plus(
                sent, 
                add_special_tokens=True,  
                max_length=512,  
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  
                return_tensors='pt',
            )
            bert_input_id = bert_encoded_dict['input_ids']
            bert_input_emo = torch.from_numpy(emo).unsqueeze(0)
            bert_attention_mask = bert_encoded_dict['attention_mask']
            label = self.csv_frame.loc[idx, 'label']

            return {'bert_input_id': bert_input_id,'bert_imput_emo': bert_input_emo,'bert_attention_mask': bert_attention_mask,
                    'label': label}
        except Exception as e:
            return None

class Fusion_Dataset(FakedditImageDataset):
    """The text + image dataset class"""

    def __init__(self, csv_file, root_dir, transform=None):
        super(Fusion_Dataset, self).__init__(csv_file, root_dir, transform)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.csv_frame.loc[idx,'id']!=np.NaN:
            img_name = str(self.csv_frame.loc[idx, 'id']) + '.jpg'
            img_path = os.path.join(self.root_dir, img_name)
        else:
            return
        try:
            sent = self.csv_frame.loc[idx, 'clean_text']
            bert_encoded_dict = self.bert_tokenizer.encode_plus(
                sent, 
                add_special_tokens=True, 
                max_length=512,  
                padding='max_length',
                truncation=True,
                return_attention_mask=True, 
                return_tensors='pt', 
            )
            bert_input_id = bert_encoded_dict['input_ids']
            bert_attention_mask = bert_encoded_dict['attention_mask']
            img_name = str(int(self.csv_frame.loc[idx, 'id'])) + '.jpg'
            img_path = os.path.join(self.root_dir, img_name)
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            label = self.csv_frame.loc[idx, 'label']
            if self.transform:
                image = self.transform(image)
            return {'bert_input_id': bert_input_id, 'bert_attention_mask': bert_attention_mask, 'image': image,
                    'label': label}
        except Exception as e:
            return None

class Hybrid_Dataset(FakedditImageDataset):
    """The text + image dataset class"""

    def __init__(self, csv_file, root_dir, transform=None):
        super(Hybrid_Dataset, self).__init__(csv_file, root_dir, transform)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.csv_frame.loc[idx,'id']!=np.NaN:
            img_name = str(int(self.csv_frame.loc[idx, 'id'])) + '.jpg'
            img_path = os.path.join(self.root_dir, img_name)
        else:
            return
        try:
            text_sent = self.csv_frame.loc[idx, 'clean_text']
            text=cut_words_from_text(text_sent)
            title_sent = self.csv_frame.loc[idx, 'clean_title']
            emo=extract_publisher_emotion(text_sent,text)
            bert_encoded_dict_text = self.bert_tokenizer.encode_plus(
                text_sent, 
                add_special_tokens=True, 
                max_length=512, 
                padding='max_length',
                truncation=True,
                return_attention_mask=True, 
                return_tensors='pt', 
            )
            bert_encoded_dict_title = self.bert_tokenizer.encode_plus(
                title_sent, 
                add_special_tokens=True, 
                max_length=512, 
                padding='max_length',
                truncation=True,
                return_attention_mask=True, 
                return_tensors='pt', 
            )
            bert_input_id_text = bert_encoded_dict_text['input_ids']
            bert_input_emo = torch.from_numpy(emo).unsqueeze(0)
            bert_attention_mask_text = bert_encoded_dict_text['attention_mask']
            bert_input_id_title = bert_encoded_dict_text['input_ids']
            bert_attention_mask_title = bert_encoded_dict_text['attention_mask']
            img_name = str(int(self.csv_frame.loc[idx, 'id'])) + '.jpg'
            img_path = os.path.join(self.root_dir, img_name)
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            label = self.csv_frame.loc[idx, 'label']
            if self.transform:
                image = self.transform(image)
            return {'bert_input_id': bert_input_id_text,'bert_input_id_title': bert_input_id_title, 
                    'bert_input_emo': bert_input_emo,'bert_attention_mask': bert_attention_mask_text,
                    'bert_attention_mask_title': bert_attention_mask_title, 'image': image,
                    'label': label}
        except Exception as e:
            return None

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

if __name__ == "__main__":
    fake_data = Hybrid_Dataset(csv_file='./Data/politifact_train.tsv', root_dir='./Data/politifact_images/')
