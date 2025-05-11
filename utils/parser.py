import pandas as pd
import torch

def parse_forecast_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    # Convert timestamps to datetime if not already
    df["forecast_made_at"] = pd.to_datetime(df["forecast_made_at"])
    df["forecast_for_hour"] = pd.to_datetime(df["forecast_for_hour"])

    # Group by location ID to isolate different cities
    grouped = df.groupby("location_id")

    input_tensors, target_values = [], []

    for location_id, data in grouped:
        # Ensure sorting for time continuity
        data = data.sort_values(by=["forecast_made_at", "forecast_for_hour"])

        # Find unique forecast made times in chronological order
        made_times = sorted(data["forecast_made_at"].unique())

        for i in range(len(made_times) - 11):  # ensures 12 rows
            subset_times = made_times[i:i+12]
            subset = data[data["forecast_made_at"].isin(subset_times)]

            # Pivot into 12x12 format
            mat = subset.pivot(index="forecast_made_at", columns="forecast_for_hour", values="temperature")
            mat = mat.sort_index().ffill(axis=1).ffill().iloc[:12, :12]
            if mat.isnull().values.any():
                continue  # Skip incomplete matrices

            if mat.shape == (12, 12):
                input_tensor = torch.tensor(mat.values, dtype=torch.float32)
                target_seq = mat.iloc[-1, -1]
                target_tensor = torch.tensor(target_seq, dtype=torch.float32)
                input_tensors.append(input_tensor)
                target_values.append(target_tensor)

    return input_tensors, target_values
