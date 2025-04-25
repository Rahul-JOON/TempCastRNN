import argparse
from utils.downloader import download_csv
from utils.parser import parse_forecast_csv
from scripts.train import train_model  # Assuming your train.py has a callable train_model function

def main():
    """
    Main function to download the CSV data, parse it, and train the model.
    This function is designed to be run from the command line with the following arguments:
        --city, --start, and --end.

        example usage:
        python model.py --city New_York --start 2023-01-01 --end 2023-01-31    

    The function will download the data from a specified URL, parse it into a format suitable for training,
    and then train a model using the parsed data.

    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    try:
        download_csv(args.city, args.start, args.end)
    except Exception as e:
        print(f"Error downloading CSV: {e}")
        return
    
    X, y = parse_forecast_csv("data/raw/raw_forecast.csv")

    train_model(X, y)

if __name__ == "__main__":
    main()
