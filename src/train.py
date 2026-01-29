import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import random
from src.data_loader import get_dataloaders
from src.model import Encoder, Decoder, Seq2Seq
from src.evaluate import calculate_bleu, calculate_cer, calculate_perplexity

def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # trg = [batch size, trg len]
        # output = [batch size, trg len, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device, tgt_tokenizer):
    model.eval()
    epoch_loss = 0
    
    all_refs = []
    all_cands = []
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            
            output = model(src, trg, 0) # turn off teacher forcing
            
            # Loss calculation
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, trg_flat)
            epoch_loss += loss.item()
            
            # Decoding for metrics
            # output = [batch size, trg len, output dim]
            top1 = output.argmax(2) 
            # top1 = [batch size, trg len]
            
            for j in range(top1.shape[0]):
                cand_indices = top1[j].tolist()
                ref_indices = trg[j].tolist()
                
                cand = tgt_tokenizer.decode(cand_indices)
                ref = tgt_tokenizer.decode(ref_indices)
                
                all_cands.append(cand)
                all_refs.append(ref)
                
    bleu = calculate_bleu(all_refs, all_cands)
    cer = calculate_cer(all_refs, all_cands)
    
    return epoch_loss / len(iterator), bleu, cer

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def run_experiment(config, dataset_path, device, experiment_name):
    print(f"\n{'='*20} Running {experiment_name} {'='*20}")
    print(f"Config: {config}")
    
    train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer = get_dataloaders(dataset_path, batch_size=config['batch_size'])
    
    INPUT_DIM = src_tokenizer.vocab_size
    OUTPUT_DIM = tgt_tokenizer.vocab_size
    
    enc = Encoder(INPUT_DIM, config['emb_dim'], config['hid_dim'], config['enc_layers'], config['dropout'])
    dec = Decoder(OUTPUT_DIM, config['emb_dim'], config['hid_dim'], config['dec_layers'], config['dropout'])
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.char2idx['<PAD>'])
    
    best_valid_loss = float('inf')
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, config['clip'], device)
        valid_loss, valid_bleu, valid_cer = evaluate(model, val_loader, criterion, device, tgt_tokenizer)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'checkpoints/{experiment_name}_best.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | BLEU: {valid_bleu:.3f} | CER: {valid_cer:.3f}')
        
        # Qualitative check every epoch
        model.eval()
        with torch.no_grad():
            # Get a random sample from validation
            src, trg = next(iter(val_loader))
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, 0) # turn off teacher forcing
            top1 = output.argmax(2)
            
            # Decode first sentence in batch
            src_sent = src_tokenizer.decode(src[0].tolist())
            trg_sent = tgt_tokenizer.decode(trg[0].tolist())
            pred_sent = tgt_tokenizer.decode(top1[0].tolist())
            
            print(f"\tExample:")
            print(f"\tSrc: {src_sent}")
            print(f"\tTrg: {trg_sent}")
            print(f"\tPrd: {pred_sent}")
        
    # Test Evaluation
    model.load_state_dict(torch.load(f'checkpoints/{experiment_name}_best.pt'))
    test_loss, test_bleu, test_cer = evaluate(model, test_loader, criterion, device, tgt_tokenizer)
    print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU: {test_bleu:.3f} | Test CER: {test_cer:.3f}')
    
    return test_bleu, test_cer

if __name__ == "__main__":
    dataset_path = '/Users/haseebabbas/Documents/NLP/dataset'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Experiment 1: Baseline
    config1 = {
        'emb_dim': 256,
        'hid_dim': 512,
        'enc_layers': 2,
        'dec_layers': 4,
        'dropout': 0.3,
        'lr': 1e-3,
        'batch_size': 128,
        'epochs': 5, # Increased for better convergence
        'clip': 1
    }
    
    # Experiment 2: Smaller Hidden Size
    config2 = config1.copy()
    config2['hid_dim'] = 256
    
    # Experiment 3: Higher Dropout
    config3 = config1.copy()
    config3['dropout'] = 0.5
    
    run_experiment(config1, dataset_path, device, 'exp1_baseline')
    run_experiment(config2, dataset_path, device, 'exp2_small_hidden')
    run_experiment(config3, dataset_path, device, 'exp3_high_dropout')
