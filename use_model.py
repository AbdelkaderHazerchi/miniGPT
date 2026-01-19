import torch
import os
from model import MiniGPT
from tokenizer import FastBPETokenizer
import platform

# ----------------------
# Utilities
# ----------------------
def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def list_folders(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    if not folders:
        print("[ERROR] No folders found!")
        return []
    for i, f in enumerate(folders):
        print(f"{i+1}. {f}")
    return folders

def list_models(path):
    files = [f for f in os.listdir(path) if f.endswith(".pt")]
    if not files:
        print("[ERROR] No model files found in this folder!")
        return []
    files.sort(key=lambda f: os.path.getmtime(os.path.join(path, f)), reverse=True)
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")
    return files

# ----------------------
# Device
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Step 1: Select model folder
# ----------------------
models_root = "models"
while True:
    clear_screen()
    print("[*] Available model folders:")
    folders = list_folders(models_root)
    if not folders:
        exit()
    choice = input("Enter folder number to enter (or 'quit' to exit): ").strip()
    if choice.lower() == "quit":
        exit()
    if choice.isdigit() and 1 <= int(choice) <= len(folders):
        folder_selected = folders[int(choice)-1]
        folder_path = os.path.join(models_root, folder_selected)
        break

# ----------------------
# Step 2: Select model in folder
# ----------------------
while True:
    clear_screen()
    print(f"[*] Models in folder '{folder_selected}':")
    files = list_models(folder_path)
    if not files:
        input("Press Enter to go back...")
        folder_selected = None
        break
    choice = input("Enter model number to load (or 'back' to go back): ").strip()
    if choice.lower() == "back":
        # Go back to folder selection
        while True:
            clear_screen()
            print("[*] Available model folders:")
            folders = list_folders(models_root)
            choice2 = input("Enter folder number to enter (or 'quit' to exit): ").strip()
            if choice2.lower() == "quit":
                exit()
            if choice2.isdigit() and 1 <= int(choice2) <= len(folders):
                folder_selected = folders[int(choice2)-1]
                folder_path = os.path.join(models_root, folder_selected)
                break
        continue
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        model_file = files[int(choice)-1]
        break

# ----------------------
# Step 3: Load model
# ----------------------
clear_screen()
print(f"Loading model '{model_file}'...")
model_path = os.path.join(folder_path, model_file)
checkpoint = torch.load(model_path, map_location=device)

# Setup tokenizer
tokenizer_path = os.path.join(folder_path, "tokenizer.json")
if os.path.exists(tokenizer_path):
    tokenizer = FastBPETokenizer()
    tokenizer.load(tokenizer_path)
    print("[OK] Tokenizer loaded from json file.")
else:
    print("[WARNING] tokenizer.json not found. Using vocab from checkpoint.")
    tokenizer = FastBPETokenizer(vocab_size=5000)

# Model config
config = checkpoint["config"]
embed_dim = config["embed_dim"]
num_heads = config["num_heads"]
num_layers = config["num_layers"]
max_seq_len = config["max_seq_len"]

vocab = checkpoint["vocab"]
# تحديث tokenizer بـ vocab من checkpoint
tokenizer.tokenizer.token_to_id = vocab
tokenizer.token_to_id = vocab  # تحديث token_to_id الخاص به
tokenizer.id_to_token = {v:k for k,v in vocab.items()}  # بناء id_to_token
tokenizer.tokenizer.id_to_token = tokenizer.id_to_token  # نسخ لـ tokenizer أيضاً
vocab_size = len(vocab)

model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    max_seq_len=max_seq_len
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"[OK] Model '{model_file}' is ready!")

# ----------------------
# Step 4: Text generation function
# ----------------------
def generate_text(model, tokenizer, prompt="", max_len=5000, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) == 0:
        input_ids = [0]  # استخدام <pad> كـ token البداية
    generated = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(generated)
            if logits.size(1) == 0:
                break
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)

    text_out = tokenizer.decode(generated[0].tolist())
    return text_out

# ----------------------
# Step 5: Choose mode
# ----------------------
while True:
    clear_screen()
    print("Select mode:")
    print("1. Auto-generate text (press Enter to generate)")
    print("2. Chat with model (enter prompts)")
    print("Type 'back' to select another model or 'quit' to exit.")
    mode_choice = input("Choice: ").strip().lower()

    if mode_choice == "/quit":
        exit()
    elif mode_choice == "/back":
        # Go back to model selection
        exec(open(__file__).read())
        exit()
    elif mode_choice == "1" or mode_choice == "":
        # Auto-generate mode
        print("Press Enter to generate text, 'quit' to exit, 'back' to go back")
        while True:
            user_input = input()
            if user_input.lower() == "/quit":
                exit()
            if user_input.lower() == "/back":
                break
            text = generate_text(model, tokenizer, prompt="", max_len=50)
            print("\n[TEXT] Generated text:")
            print(text)
            print("\n---\nPress Enter again or type 'back'/'quit'")
    elif mode_choice == "2":
        # Chat mode
        print("Enter 'quit' to exit, 'back' to go back")
        while True:
            user_prompt = input("You: ")
            if user_prompt.lower() == "/quit":
                exit()
            if user_prompt.lower() == "/back":
                break
            text = generate_text(model, tokenizer, prompt=user_prompt, max_len=50)
            print(f"Model: {text}")
