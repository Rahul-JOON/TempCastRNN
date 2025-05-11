import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import os
import json

# 1. Simulated Dataset (replace with CSV loader later)
class ForecastDataset(Dataset):
    def __init__(self, X, y):
        self.inputs = X
        self.targets = y

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx].view(1)

# 2. Transformer-based Model
class ForecastTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super(ForecastTransformer, self).__init__()
        self.embedding = nn.Linear(12, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        x = self.embedding(x)        # (B, 12, d_model)
        x = self.transformer(x, mask=mask)
        return self.fc(x[:, -1])        # Take last timestep

# 3. Mask Generator for Transformer
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

# 4. Training Function

def train_model(X, y, batch_size=32, epochs=300, val_loss_threshold=0.1, save_path="models/best_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(ForecastDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ForecastDataset(X_val, y_val), batch_size=batch_size)

    model = ForecastTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.SmoothL1Loss()

    try:
        with open("models/best_model.json", "r") as f:
            best_val_loss = json.load(f)["val_loss"]
    except FileNotFoundError:
        best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_batch = y_batch.view(-1, 1)
            optimizer.zero_grad()
            mask = generate_square_subsequent_mask(x_batch.size(1)).to(device)
            output = model(x_batch, mask=mask)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_batch = y_batch.view(-1, 1)
                mask = generate_square_subsequent_mask(x_batch.size(1)).to(device)
                output = model(x_batch, mask=mask)
                loss = criterion(output, y_batch)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model (val loss = {best_val_loss:.4f}) to '{save_path}'")

            # Save metadata
            meta = {
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "model_config": {
                    "d_model": model.embedding.out_features,
                    "nhead": model.transformer.layers[0].self_attn.num_heads,
                    "num_layers": len(model.transformer.layers)
                },
                "training_config": {
                    "batch_size": batch_size,
                    "weight_decay": optimizer.param_groups[0]['weight_decay'],
                    "lr": optimizer.param_groups[0]['lr'],
                    "scheduler": "CosineAnnealingLR"
                },
                "dataset_info": {
                    "city": "City_Name",   # <-- replace with actual city name if available
                    "train_size": len(X_train),
                    "val_size": len(X_val)
                }
            }

            meta_path = save_path.replace(".pth", ".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)

        # Early stopping condition
        if avg_val_loss < val_loss_threshold:
            print(f"\n✅ Early stopping at epoch {epoch+1}: Validation loss {avg_val_loss:.4f} < threshold {val_loss_threshold}")
            break

    # Plot both training and validation loss
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(f"logs/train_logs/{timestamp}", exist_ok=True)
    os.chdir(f"logs/train_logs/{timestamp}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'loss_plot_{timestamp}.png')
    # plt.show()

    # Save CSV
    with open(f'training_loss_{timestamp}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
        for i in range(len(train_losses)):
            writer.writerow([i+1, train_losses[i], val_losses[i]])

    return model