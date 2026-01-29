import torch
from src.model import Encoder, Decoder, Seq2Seq
from src.data_loader import Tokenizer, load_data
import os

def verify_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Tokenizers
    dataset_path = '/Users/haseebabbas/Documents/NLP/dataset'
    pairs = load_data(dataset_path)
    
    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()
    src_tokenizer.build_vocab([p[0] for p in pairs])
    tgt_tokenizer.build_vocab([p[1] for p in pairs])
    
    # Model Config (Baseline)
    INPUT_DIM = src_tokenizer.vocab_size
    OUTPUT_DIM = tgt_tokenizer.vocab_size
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    ENC_LAYERS = 2
    DEC_LAYERS = 4
    ENC_DROPOUT = 0.3
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_LAYERS, ENC_DROPOUT)
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    checkpoint_path = 'checkpoints/exp1_baseline_best.pt'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Model checkpoint not found!")
        return

    model.eval()
    
    test_sentences = [
        "دل سے اتر جائے گا",
        "زندگی تیری عطا ہے",
        "ظالم اب کے بھی نہ روئے گا"
    ]
    
    for sentence in test_sentences:
        tokens = [src_tokenizer.char2idx.get(c, src_tokenizer.char2idx['<UNK>']) for c in sentence]
        tokens = [src_tokenizer.char2idx['<SOS>']] + tokens + [src_tokenizer.char2idx['<EOS>']]
        
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            hidden, cell = model.encoder(src_tensor)
        
        trg_indexes = [tgt_tokenizer.char2idx['<SOS>']]
        
        for i in range(100):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            
            if pred_token == tgt_tokenizer.char2idx['<EOS>']:
                break
                
        trg_tokens = [tgt_tokenizer.idx2char[i] for i in trg_indexes]
        translation = "".join(trg_tokens[1:-1])
        
        print(f"Urdu: {sentence}")
        print(f"Roman: {translation}")
        print("-" * 30)

if __name__ == "__main__":
    verify_inference()
