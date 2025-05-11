import argparse
import os
import pandas as pd
from utils.downloader import download_csv
from utils.parser import parse_forecast_csv
from scripts.train import train_model
from scripts.evaluate import evaluate
from dotenv import load_dotenv

def main():
    """
    Main function to download the CSV data, parse it, and train/evaluate the model.
    This function is designed to be run from the command line with the following arguments:
        --cities(array of cities), --start(date), --end(date), --url(api route), --mode(train/evaluate)
    
    Example command to train the model:
        python model.py --cities New_York Najafgarh --start 2023-01-01 --end 2023-01-31 --url https://example.com/forecast.csv --mode train

    Example command to evaluate the model:
        python model.py --cities New_York Najafgarh --start 2023-01-01 --end 2023-01-31 --url https://example.com/forecast.csv --mode evaluate

    This script will download the forecast data for the specified cities, parse the CSV files, and train or evaluate the model based on the provided mode.

    """
    
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", required=True, help="List of city names")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    load_dotenv()
    parser.add_argument("--url", default=os.getenv("DOWNLOAD_URL"), help="URL to download CSV data")
    parser.add_argument("--mode", default="train", help="Mode to run the script in (train/evaluate)")
    args = parser.parse_args()


    if args.mode == "evaluate":
        # evaluate the model
        evaluate(args.cities, args.start, args.end, args.url)
        return
    
    elif args.mode == "train":
        # Download the CSV data for the specified cities
        # and parse it into a DataFrame
        os.makedirs("data/raw", exist_ok=True)
        all_dfs = []

        for city in args.cities:
            city = city.replace("_", " ")
            city_file = f"data/raw/{city}_forecast.csv"
            try:
                download_csv(city, args.start, args.end, args.url, save_path=city_file)
                df = pd.read_csv(city_file)
                df["city"] = city  # Add a city column for traceability
                all_dfs.append(df)
            except Exception as e:
                print(f"⚠️ Skipped {city}: {e}")

        # Combine all DataFrames into one and save to CSV
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv("data/raw/raw_forecast.csv", index=False)
            print("📊 Combined CSV saved to data/raw/raw_forecast.csv")
        else:
            print("❌ No data was successfully downloaded.")
        
        X, y = parse_forecast_csv("data/raw/raw_forecast.csv")

        # Train the model
        train_model(X, y)

    else:
        # Invalid mode
        print("❌ Invalid mode. Use 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()
