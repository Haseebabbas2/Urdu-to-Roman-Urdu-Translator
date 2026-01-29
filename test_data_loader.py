from src.data_loader import get_dataloaders
import torch

def test_loader():
    dataset_path = '/Users/haseebabbas/Documents/NLP/dataset'
    try:
        train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer = get_dataloaders(dataset_path, batch_size=4)
        
        print("Data loading successful!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Check a batch
        src_batch, tgt_batch = next(iter(train_loader))
        print(f"Source batch shape: {src_batch.shape}")
        print(f"Target batch shape: {tgt_batch.shape}")
        
        # Decode first sample
        src_sample = src_batch[0].tolist()
        tgt_sample = tgt_batch[0].tolist()
        
        print(f"Source (Indices): {src_sample}")
        print(f"Target (Indices): {tgt_sample}")
        
        print(f"Source (Decoded): {src_tokenizer.decode(src_sample)}")
        print(f"Target (Decoded): {tgt_tokenizer.decode(tgt_sample)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_loader()
