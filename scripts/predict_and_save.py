import torch
import pandas as pd
import numpy as np
from model import ForecastTransformer, generate_square_subsequent_mask

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ForecastTransformer()
model.load_state_dict(torch.load('models/forecast_model.pt', map_location=device))
model.to(device)
model.eval()

# Load new 12x1 forecast inputs (example simulation, replace with actual input)
new_inputs = torch.randn(10, 12, 1)  # Simulate 10 new samples of 12x1

# Pad inputs to 12x12 with masked values (e.g., -inf or 0) for transformer input compatibility
masked_inputs = torch.full((10, 12, 12), float('-inf'))  # or 0 if masking internally
masked_inputs[:, :, 0] = new_inputs.squeeze(-1)  # Fill first column with actual forecast

# Forward pass
with torch.no_grad():
    masked_inputs = masked_inputs.to(device)
    mask = generate_square_subsequent_mask(masked_inputs.size(1)).to(device)
    predictions = model(masked_inputs, src_mask=mask).cpu().numpy().squeeze()

# Save predictions to CSV
df = pd.DataFrame(predictions, columns=["predicted_temp"])
df.to_csv("predictions/output_predictions.csv", index=False)
print("Predictions saved to predictions/output_predictions.csv")
