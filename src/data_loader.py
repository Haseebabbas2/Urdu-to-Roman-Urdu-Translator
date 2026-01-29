import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import re

class Tokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        
    def build_vocab(self, text_list):
        chars = set()
        for text in text_list:
            chars.update(list(text))
        
        # Sort for determinism
        chars = sorted(list(chars))
        
        # Add special tokens first
        for token in self.special_tokens:
            self.char2idx[token] = len(self.char2idx)
            self.idx2char[len(self.idx2char)] = token
            
        # Add characters
        for char in chars:
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)
                self.idx2char[len(self.idx2char)] = char
                
        self.vocab_size = len(self.char2idx)
        
    def encode(self, text):
        return [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]
    
    def decode(self, indices):
        return ''.join([self.idx2char.get(idx, '') for idx in indices if idx not in [self.char2idx['<PAD>'], self.char2idx['<SOS>'], self.char2idx['<EOS>']]])

class UrduRomanDataset(Dataset):
    def __init__(self, pairs, src_tokenizer, tgt_tokenizer):
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        
        src_indices = [self.src_tokenizer.char2idx['<SOS>']] + \
                      self.src_tokenizer.encode(src_text) + \
                      [self.src_tokenizer.char2idx['<EOS>']]
                      
        tgt_indices = [self.tgt_tokenizer.char2idx['<SOS>']] + \
                      self.tgt_tokenizer.encode(tgt_text) + \
                      [self.tgt_tokenizer.char2idx['<EOS>']]
                      
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    
    return src_padded, tgt_padded

def load_data(dataset_path):
    pairs = []
    
    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        if 'ur' in dirs and 'en' in dirs:
            ur_path = os.path.join(root, 'ur')
            en_path = os.path.join(root, 'en')
            
            ur_files = sorted(os.listdir(ur_path))
            en_files = sorted(os.listdir(en_path))
            
            # Create a map for english files for faster lookup
            en_files_map = {f: f for f in en_files}
            
            for ur_file in ur_files:
                if ur_file in en_files_map:
                    try:
                        with open(os.path.join(ur_path, ur_file), 'r', encoding='utf-8') as f:
                            ur_lines = f.readlines()
                        with open(os.path.join(en_path, ur_file), 'r', encoding='utf-8') as f:
                            en_lines = f.readlines()
                            
                        # Align lines (assuming 1-to-1 correspondence)
                        if len(ur_lines) == len(en_lines):
                            for u, e in zip(ur_lines, en_lines):
                                u = u.strip()
                                e = e.strip()
                                if u and e: # Skip empty lines
                                    pairs.append((u, e))
                    except Exception as e:
                        print(f"Error reading {ur_file}: {e}")
                        
    return pairs

def get_dataloaders(dataset_path, batch_size=32, split_ratios=(0.5, 0.25, 0.25)):
    pairs = load_data(dataset_path)
    print(f"Total pairs found: {len(pairs)}")
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Split
    n = len(pairs)
    train_end = int(n * split_ratios[0])
    val_end = int(n * (split_ratios[0] + split_ratios[1]))
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    # Build Tokenizers
    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()
    
    src_texts = [p[0] for p in pairs]
    tgt_texts = [p[1] for p in pairs]
    
    src_tokenizer.build_vocab(src_texts)
    tgt_tokenizer.build_vocab(tgt_texts)
    
    print(f"Source Vocab Size: {src_tokenizer.vocab_size}")
    print(f"Target Vocab Size: {tgt_tokenizer.vocab_size}")
    
    # Create Datasets
    train_dataset = UrduRomanDataset(train_pairs, src_tokenizer, tgt_tokenizer)
    val_dataset = UrduRomanDataset(val_pairs, src_tokenizer, tgt_tokenizer)
    test_dataset = UrduRomanDataset(test_pairs, src_tokenizer, tgt_tokenizer)
    
    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer
