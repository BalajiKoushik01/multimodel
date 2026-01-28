import os
import sys
import shutil

def download_mobileodt():
    # check for kaggle
    try:
        import kaggle
    except ImportError:
        print("Kaggle library not installed. Installing...")
        os.system(f"{sys.executable} -m pip install kaggle")
        try:
            import kaggle
        except ImportError:
            print("Failed to install kaggle. Please install manually: pip install kaggle")
            return

    # check for credentials
    kaggle_dir = os.path.expanduser('~/.kaggle')
    if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
        print(f"Error: kaggle.json not found in {kaggle_dir}")
        print("Please download your API token from Kaggle -> Account -> Create New API Token")
        print(f"and save it to {os.path.join(kaggle_dir, 'kaggle.json')}")
        return

    print("Kaggle credentials found.")
    
    dataset_name = "intel-mobileodt-cervical-cancer-screening"
    target_dir = os.path.join("data", "mobileodt")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Downloading {dataset_name} to {target_dir}...")
    
    # Use kaggle python api directly
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    
    # Needs authentication
    try:
        api.authenticate()
        print("Authenticated successfully.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    try:
        # competition download
        api.competition_download_files(dataset_name, path=target_dir, quiet=False)
    except Exception as e:
        print(f"Download failed: {e}")
        if "403" in str(e):
            print("Action Required: You must accept the competition rules.")
            print(f"Go to https://www.kaggle.com/c/{dataset_name}/rules and click 'I Understand and Accept'.")
        return

    print("Download complete. Extracting files...")
    # It likely downloaded a zip. Find it.
    for fname in os.listdir(target_dir):
        if fname.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(os.path.join(target_dir, fname), 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"Extracted {fname}")
        elif fname.endswith('.7z'):
            print(f"Found .7z file: {fname}. Please install 7zip or 'pip install py7zr' to extract.")
            try:
                import py7zr
                with py7zr.SevenZipFile(os.path.join(target_dir, fname), mode='r') as z:
                    z.extractall(path=target_dir)
                print(f"Extracted {fname}")
            except ImportError:
                print("py7zr not installed. Run 'pip install py7zr' and try again.")
    
    print("MobileODT setup check complete.")

if __name__ == "__main__":
    download_mobileodt()
