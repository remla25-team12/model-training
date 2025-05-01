import os
import requests

def download_data():
    # Link to the training dataset from the Sentiment Analysis repository
    url = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "a1_RestaurantReviews_HistoricDump.tsv")

    response = requests.get(url)
    if response.status_code == 200:
         with open(output_file, "wb") as f:
            f.write(response.content)
            print("Dataset downloaded successfully.")
    else:
        print(f"Failed to download the dataset with status code: {response.status_code}")
    
if __name__ == "__main__":
    download_data()

    