from src.data_loader import load_data, Tokenizer

def check_vocab():
    dataset_path = '/Users/haseebabbas/Documents/NLP/dataset'
    pairs = load_data(dataset_path)
    
    src_tokenizer = Tokenizer()
    src_tokenizer.build_vocab([p[0] for p in pairs])
    
    input_text = "و علیکم السلام"
    print(f"Input: {input_text}")
    
    print("Character check:")
    for char in input_text:
        if char in src_tokenizer.char2idx:
            print(f"'{char}': Found (ID: {src_tokenizer.char2idx[char]})")
        else:
            print(f"'{char}': NOT FOUND (Will be <UNK>)")

if __name__ == "__main__":
    check_vocab()
