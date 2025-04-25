import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. Simulated Dataset (replace with CSV loader later)
class ForecastDataset(Dataset):
    def __init__(self, X, y):
        self.inputs = X
        self.targets = y

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# 2. Transformer-based Model
class ForecastTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super(ForecastTransformer, self).__init__()
        self.embedding = nn.Linear(12, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        x = self.embedding(x)        # (B, 12, d_model)
        x = x.permute(1, 0, 2)       # (12, B, d_model)
        x = self.transformer(x, mask=mask)
        return self.fc(x[-1])        # Take last timestep

# 3. Mask Generator for Transformer
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

# 4. Training Function

def train_model(X, y, batch_size=32, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ForecastDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ForecastTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=10e-10, weight_decay=10e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.L1Loss()

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            mask = generate_square_subsequent_mask(x_batch.size(1)).to(device)
            output = model(x_batch, mask=mask)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o', label='MAE Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model
