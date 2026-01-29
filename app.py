import streamlit as st
import torch
from src.model import Encoder, Decoder, Seq2Seq
from src.data_loader import Tokenizer
import os

# Page Config
st.set_page_config(
    page_title="Urdu to Roman Urdu NMT",
    page_icon="🌙",
    layout="centered"
)

# Custom CSS for Aesthetic Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1 {
        font-weight: 700;
        color: #a8d0e6;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #b8b8d0;
    }
    
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.2rem;
        direction: rtl;
    }
    
    .stTextArea textarea:focus {
        border-color: #a8d0e6;
        box-shadow: 0 0 10px rgba(168, 208, 230, 0.3);
    }
    
    .stButton button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(75, 108, 183, 0.4);
    }
    
    .result-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .result-text {
        font-size: 1.5rem;
        font-weight: 600;
        color: #a8d0e6;
    }
    
    .footer {
        text-align: center;
        margin-top: 4rem;
        color: #6c757d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_model_and_tokenizers():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load Tokenizers (Rebuild them as they are not saved separately, 
    # in a real app we should save/load vocab pickle)
    # For now, we rebuild from dataset (fast enough for this demo)
    from src.data_loader import load_data
    dataset_path = '/Users/haseebabbas/Documents/NLP/dataset'
    pairs = load_data(dataset_path)
    
    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()
    src_tokenizer.build_vocab([p[0] for p in pairs])
    tgt_tokenizer.build_vocab([p[1] for p in pairs])
    
    # Model Config (Must match training config)
    # Using Baseline config
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
    
    # Load weights if exist
    checkpoint_path = 'checkpoints/exp1_baseline_best.pt'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        st.warning("Model checkpoint not found. Using initialized weights.")
        
    model.eval()
    return model, src_tokenizer, tgt_tokenizer, device

def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, device, max_len=100):
    model.eval()
    
    tokens = [src_tokenizer.char2idx.get(c, src_tokenizer.char2idx['<UNK>']) for c in sentence]
    tokens = [src_tokenizer.char2idx['<SOS>']] + tokens + [src_tokenizer.char2idx['<EOS>']]
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device) # [1, src_len]
    
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    
    trg_indexes = [tgt_tokenizer.char2idx['<SOS>']]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == tgt_tokenizer.char2idx['<EOS>']:
            break
            
    trg_tokens = [tgt_tokenizer.idx2char[i] for i in trg_indexes]
    
    # Remove SOS and EOS
    return "".join(trg_tokens[1:-1])

# Main UI
st.markdown('<div class="main-header"><h1>🌙 Urdu to Roman Urdu NMT</h1><p class="subtitle">Neural Machine Translation with BiLSTM</p></div>', unsafe_allow_html=True)

# Load resources
try:
    model, src_tokenizer, tgt_tokenizer, device = load_model_and_tokenizers()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ✍️ Urdu Input")
        urdu_text = st.text_area("Enter Urdu text here...", height=150, placeholder="یہاں اردو لکھیں...")
        
    with col2:
        st.markdown("### 🔤 Roman Urdu Output")
        if st.button("Translate", use_container_width=True):
            if urdu_text:
                with st.spinner("Translating..."):
                    translation = translate_sentence(model, urdu_text, src_tokenizer, tgt_tokenizer, device)
                    st.markdown(f'<div class="result-box"><p class="result-text">{translation}</p></div>', unsafe_allow_html=True)
            else:
                st.info("Please enter some text to translate.")
        else:
             st.markdown('<div class="result-box" style="opacity:0.5"><p class="result-text">Translation will appear here</p></div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading model: {e}")

st.markdown('<div class="footer">Project by Haseeb Abbas | Powered by PyTorch & Streamlit</div>', unsafe_allow_html=True)
