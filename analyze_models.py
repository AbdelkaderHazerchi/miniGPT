import torch
import os
from model import MiniGPT
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
        file_size = os.path.getsize(os.path.join(path, f)) / (1024 * 1024)
        print(f"{i+1}. {f} ({file_size:.2f} MB)")
    return files

def format_size(size_bytes):
    """تحويل الحجم بالبايتات إلى صيغة مقروءة"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def count_parameters(model):
    """حساب عدد المعاملات في النموذج"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_model_info(model_path, checkpoint, config, vocab_size):
    """طباعة معلومات شاملة عن النموذج"""
    
    print("\n" + "=" * 70)
    print("MODEL INFORMATION & ANALYSIS".center(70))
    print("=" * 70)
    
    # 1. معلومات الملف
    print("\n[FILE INFORMATION]")
    print(f"  Path: {model_path}")
    file_size = os.path.getsize(model_path)
    print(f"  File Size: {format_size(file_size)}")
    
    # 2. معلومات Checkpoint
    print("\n[CHECKPOINT INFORMATION]")
    print(f"  Checkpoint Keys: {', '.join(checkpoint.keys())}")
    if 'epoch' in checkpoint:
        print(f"  Trained Epochs: {checkpoint['epoch']}")
    
    # 3. إعدادات النموذج
    print("\n[MODEL CONFIGURATION]")
    print(f"  Embedding Dimension: {config['embed_dim']}")
    print(f"  Number of Heads (Attention): {config['num_heads']}")
    print(f"  Number of Layers: {config['num_layers']}")
    print(f"  Max Sequence Length: {config['max_seq_len']}")
    print(f"  Vocabulary Size: {vocab_size}")
    
    # 4. بناء النموذج وحساب المعاملات
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    total_params, trainable_params = count_parameters(model)
    
    print("\n[PARAMETER STATISTICS]")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # 5. توزيع المعاملات حسب الطبقات
    print("\n[LAYER BREAKDOWN]")
    
    layer_params = {}
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0] if '.' in name else name
        if layer_name not in layer_params:
            layer_params[layer_name] = 0
        layer_params[layer_name] += param.numel()
    
    for layer_name, param_count in sorted(layer_params.items(), key=lambda x: x[1], reverse=True):
        percentage = (param_count / total_params) * 100
        print(f"  {layer_name}: {param_count:,} ({percentage:.1f}%)")
    
    # 6. حساب حجم الذاكرة المتوقع
    print("\n[MEMORY REQUIREMENTS]")
    model_memory = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"  Model Size (float32): ~{model_memory:.2f} MB")
    
    # مع activation memory (تقريبي)
    batch_size = 1
    seq_len = config['max_seq_len']
    activation_memory = (batch_size * seq_len * config['embed_dim'] * 4) / (1024 * 1024)
    total_memory = model_memory + activation_memory
    print(f"  Estimated RAM (inference): ~{total_memory:.2f} MB")
    
    # 7. معلومات التدريب
    print("\n[TRAINING INFORMATION]")
    if 'optimizer_state_dict' in checkpoint:
        print(f"  Optimizer State: Saved")
    else:
        print(f"  Optimizer State: Not saved")
    
    if 'epoch' in checkpoint:
        print(f"  Checkpoint Epoch: {checkpoint['epoch']}")
    
    # 8. حجم القاموس والرموز
    print("\n[VOCABULARY INFORMATION]")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Vocab Storage Size: ~{(vocab_size * 4) / 1024:.2f} KB (estimated)")
    
    # إحصائيات إضافية عن vocab من checkpoint
    if 'vocab' in checkpoint:
        vocab = checkpoint['vocab']
        special_tokens = [k for k in vocab.keys() if k.startswith('<')]
        print(f"  Special Tokens: {len(special_tokens)} ({', '.join(sorted(special_tokens))})")
    
    # 9. معلومات الجهاز
    print("\n[DEVICE INFORMATION]")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 10. ملخص الأداء المتوقع
    print("\n[EXPECTED PERFORMANCE]")
    flops = total_params * seq_len * 2  # تقريبي للحسابات
    print(f"  Approximate FLOPs per token: {flops / 1e9:.2f} B")
    print(f"  Model Complexity: {'Simple' if total_params < 1e6 else 'Medium' if total_params < 10e6 else 'Complex'}")
    
    print("\n" + "=" * 70 + "\n")
    
    return model

# ----------------------
# Main Program
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
    choice = input("\nEnter folder number to enter (or 'quit' to exit): ").strip()
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
    choice = input("\nEnter model number to analyze (or 'back' to go back): ").strip()
    if choice.lower() == "back":
        # Go back to folder selection
        while True:
            clear_screen()
            print("[*] Available model folders:")
            folders = list_folders(models_root)
            choice2 = input("\nEnter folder number to enter (or 'quit' to exit): ").strip()
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
# Step 3: Load and analyze model
# ----------------------
clear_screen()
print(f"Loading model '{model_file}'...")
model_path = os.path.join(folder_path, model_file)

try:
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    vocab_size = len(vocab)
    
    # عرض المعلومات
    model = print_model_info(model_path, checkpoint, config, vocab_size)
    
except Exception as e:
    print(f"\n[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()

input("Press Enter to exit...")
