import torch
from src.model import Encoder, Decoder, Seq2Seq
from src.data_loader import get_dataloaders

def test_model():
    dataset_path = '/Users/haseebabbas/Documents/NLP/dataset'
    train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer = get_dataloaders(dataset_path, batch_size=4)
    
    # Hyperparameters
    INPUT_DIM = src_tokenizer.vocab_size
    OUTPUT_DIM = tgt_tokenizer.vocab_size
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    ENC_LAYERS = 2
    DEC_LAYERS = 4
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    print("Model created successfully!")
    print(model)
    
    # Get a batch
    src, trg = next(iter(train_loader))
    src = src.to(device)
    trg = trg.to(device)
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {trg.shape}")
    
    # Forward pass
    output = model(src, trg)
    
    print(f"Output shape: {output.shape}")
    # Output should be [batch size, trg len, output dim]
    
    assert output.shape == (4, trg.shape[1], OUTPUT_DIM)
    print("Forward pass successful!")

if __name__ == "__main__":
    test_model()
