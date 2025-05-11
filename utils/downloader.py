import requests

def download_csv(city: str, start: str, end: str, url, save_path: str = "data/raw/raw_forecast.csv"):
    # url = "https://forecast-dashboard.vercel.app/api/download"
    payload = {
        "city": city,
        "start_date": start,
        "end_date": end
    }

    print(f"Requesting data for {city} from {start} to {end}...")
    response = requests.get(url, params=payload)

    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Data saved to {save_path}")
    else:
        raise Exception(f"Failed to download data. Status code: {response.status_code}")
