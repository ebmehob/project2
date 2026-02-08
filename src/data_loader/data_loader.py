import os
import requests
import zipfile
import argparse
import sys
from pathlib import Path

def setup_directories(output_dir):
    """
    Creates the target directory if it doesn't exist.
    """
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"--- Data directory ready at: {target_path} ---")
    return target_path

def download_and_extract_csv_only(url, target_folder):
    """
    Downloads a zip file, extracts only CSVs to the target folder (flattening structure),
    and cleans up the zip file.
    """
    zip_path = target_folder / "temp_archive.zip"
    
    print(f"Downloading archive from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return

    print(f"Searching for CSV files inside archive...")
    extracted_count = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.endswith('.csv') and not member.startswith('__MACOSX'):
                    filename = os.path.basename(member)
                    
                    if not filename:
                        continue

                    source = zip_ref.open(member)
                    target_file_path = target_folder / filename
                    
                    with open(target_file_path, "wb") as target_file:
                        target_file.write(source.read())
                    
                    print(f"Extracted: {filename}")
                    extracted_count += 1
                    
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip archive.")
    finally:
        if zip_path.exists():
            os.remove(zip_path)
            print("Cleanup: Temporary zip removed.")
            
    if extracted_count == 0:
        print("Warning: No CSV files were found/extracted.")

if __name__ == "__main__":
    TRAIN_URL = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_train_dataset.zip"
    TEST_URL = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_inference_dataset.zip"
    
    parser = argparse.ArgumentParser(description="Download and extract datasets.")
    parser.add_argument('--output_dir', type=str, default='/data/raw', help='Path to save extracted data')
    
    args = parser.parse_args()
    
    target_folder = setup_directories(args.output_dir)
    download_and_extract_csv_only(TRAIN_URL, target_folder)
    download_and_extract_csv_only(TEST_URL, target_folder)
    
    print("\nWorkflow finished successfully.")