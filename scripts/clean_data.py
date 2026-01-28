
import os
import shutil

def clean_directory(root_dir):
    print(f"Cleaning {root_dir}...")
    
    # 1. Remove __pycache__
    for root, dirs, files in os.walk(root_dir):
        if '__pycache__' in dirs:
            p = os.path.join(root, '__pycache__')
            print(f"Removing {p}")
            shutil.rmtree(p)
            
    # 2. Check for empty folders and remove them
    # Walk bottom-up
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in dirs:
            p = os.path.join(root, name)
            if not os.listdir(p): # Empty
                print(f"Removing empty directory {p}")
                os.rmdir(p)
                
    # 3. Specific cleanup for AnnoCerv repo artifacts if they exist
    repo_dir = os.path.join(root_dir, 'annocerv_repo')
    if os.path.exists(repo_dir):
        print("Note: 'annocerv_repo' exists. Ensure you have moved data to 'annocerv_real' or keep it if needed for git.")
        
    print("Cleanup complete.")

if __name__ == "__main__":
    clean_directory("data")
