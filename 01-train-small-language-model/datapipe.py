import ast
import time
import torch
import datasets
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datasets import (
                    load_dataset, 
                    DatasetDict, 
                    Dataset
                    )
from torch.utils.data import (
                            Dataset, 
                            DataLoader, 
                            random_split
                            )
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Load data set from huggingface
data_sample = load_dataset("QuyenAnhDE/Diseases_Symptoms")

updated_data = [{'Name': item['Name'], 'Symptoms': item['Symptoms']} for item in data_sample['train']]
df = pd.DataFrame(updated_data)
df.to_csv('data.csv', index=False)

# convert to list of symptoms to array
df['Symptoms'] = df['Symptoms'].apply(lambda x: ', '.join(x.split(', ')))

# If you have an NVIDIA GPU attached, use 'cuda'
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    # If Apple Silicon, set to 'mps' - otherwise 'cpu' (not advised)
    try:
        device = torch.device('mps')
    except Exception:
        device = torch.device('cpu')

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Dataset Prep
class LanguageDataset(Dataset):
    """
    An extension of the Dataset object to:
      - Make training loop cleaner
      - Make ingestion easier from pandas df's
    """
    def __init__(self, df, tokenizer):
        self.labels = df.columns
        self.data = df.to_dict(orient='records')
        self.tokenizer = tokenizer
        x = self.fittest_max_length(df)  # Fix here
        self.max_length = x

        print("Max length set to: ", self.max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][self.labels[0]]
        y = self.data[idx][self.labels[1]]
        text = f"{x} | {y}"
        tokens = self.tokenizer.encode_plus(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        return tokens

    def fittest_max_length(self, df):  # Fix here
        """
        Smallest power of two larger than the longest term in the data set.
        Important to set up max length to speed training time.
        """
        max_length = max(len(max(df[self.labels[0]], key=len)), len(max(df[self.labels[1]], key=len)))
        x = 2
        while x < max_length: x = x * 2
        return x

# Cast the Huggingface data set as a LanguageDataset we defined above
data_sample = LanguageDataset(df, tokenizer)

train_size = int(0.8 * len(data_sample))
valid_size = len(data_sample) - train_size
train_data, valid_data = random_split(
                                    data_sample, 
                                    [
                                        train_size, 
                                        valid_size
                                    ])
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=8)
