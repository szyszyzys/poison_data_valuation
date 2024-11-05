import csv
from pathlib import Path

import requests

# Specify the path to your CSV file and the directory to save images
csv_file_path = "data/fitzpatrick17k/fitzpatrick-mod.csv"

image_save_dir = Path("data/fitzpatrick17k/images")
image_save_dir.mkdir(exist_ok=True)

# Custom headers with a User-Agent to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
}

# Open the CSV and read each row
with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        md5hash = row["md5hash"]
        url = row["url"]

        if url and md5hash:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                # Save the image with the MD5 hash as the filename
                image_filename = image_save_dir / f"{md5hash}.jpg"
                with open(image_filename, "wb") as image_file:
                    image_file.write(response.content)

                print(f"Downloaded and saved {image_filename}")

            except requests.exceptions.RequestException as e:
                print(f"Could not download image from {url}: {e}")
