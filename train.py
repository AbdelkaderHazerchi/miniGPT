import torch
import torch.nn as nn
from model import MiniGPT
from tokenizer import FastBPETokenizer  # أو tokenizer الذي بنيته
import os
import glob

# المجلد الذي يحتوي على الملفات النصية
data_folder = "data"

# إعدادات التدريب
device = "cuda"  if torch.cuda.is_available() else "cpu"

embed_dim = 128
num_heads = 4
num_layers = 2
max_seq_len = 128

batch_size = 8
epochs = 1000
lr = 2e-4
save_every = 200  # كل 100 epoch احفظ النموذج
save_dir = "models_g2"
model_name = "minigpt_model.pt"

os.makedirs(save_dir, exist_ok=True)


save_path = os.path.join(save_dir, model_name)

def save_model(model, tokenizer, epoch, optimizer, save_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "vocab": tokenizer.tokenizer.get_vocab(),
        "config": {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "max_seq_len": max_seq_len
        }
    }

    torch.save(checkpoint, save_path)
    print(f"🔹 The model has been saved in {save_path}")

def load_model(model, tokenizer, optimizer, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    tokenizer_vocab = checkpoint["vocab"]
    tokenizer.tokenizer.token_to_id = tokenizer_vocab  # استرجاع القاموس
    print(f"🔹 The model was downloaded from {load_path}")

    return model, tokenizer, optimizer, checkpoint["epoch"]

def get_latest_checkpoint(save_dir):
    files = [f for f in os.listdir(save_dir) if f.endswith(".pt")]
    if not files:
        return None
    # ترتيب الملفات حسب وقت التعديل (آخر حفظ أولاً)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(save_dir, f)), reverse=True)
    return os.path.join(save_dir, files[0])




# جلب كل الملفات النصية في المجلد
file_list = glob.glob(os.path.join(data_folder, "*.txt"))

# قراءة كل الملفات ودمجها في نص واحد
text = ""
for file_path in file_list:
    with open(file_path, "r", encoding="utf-8") as f:
        text += f.read() + "\n"  # إضافة فاصل سطر بين الملفات

# تدريب التوكينايزر على كل الملفات
tokenizer = FastBPETokenizer(vocab_size=5000)
tokenizer.train(file_list)  # تمرير كل الملفات للتوكينايزر

# حفظ التوكينايزر لاستخدامه لاحقاً
tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

vocab_size = tokenizer.tokenizer.get_vocab_size()

encoded = tokenizer.encode(text)
data = torch.tensor(encoded, dtype=torch.long)

def get_batch(data, batch_size, seq_len):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))

    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])

    return x.to(device), y.to(device)


model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    max_seq_len=max_seq_len
).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


latest_ckpt = get_latest_checkpoint(save_dir)

start_epoch = 0  # افتراضي
if latest_ckpt is not None:
    model, tokenizer, optimizer, start_epoch = load_model(
        model, tokenizer, optimizer, latest_ckpt, device
    )
    start_epoch += 1  # نبدأ من epoch التالي
    print(f"🔹 🔹 Resumption of training from epoch {start_epoch}")
else:
    print("🔹 No copies are saved, start from the beginning")


model.train()
for epoch in range(start_epoch, epochs):
    x, y = get_batch(data, batch_size, max_seq_len)

    logits = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # حفظ دوري كل save_every epoch
    if epoch % save_every == 0 and epoch != 0:
        save_model(model, tokenizer, epoch, optimizer, save_path)

print("\nTraining complete!")
final_model_path = os.path.join(save_dir, "final_model.pt")
save_model(model, tokenizer, epochs, optimizer, final_model_path)
print(f"✅ The final model has been saved in {final_model_path}")
