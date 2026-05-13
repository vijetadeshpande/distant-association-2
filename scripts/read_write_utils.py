import subprocess
import os

def copy_to_gdrive(local_path, remote_folder_name="gdrive-distant-association:distant-association-drive/"):
    """
    Copies a file or folder to a specific folder on your mounted Google Drive.
    
    # Example usage:
    # local_results = "./experiment_logs"
    # target_dir = "Research/May_Results"
    # copy_to_gdrive(local_results, target_dir)
    """
    # Your remote name from rclone config
    remote_name = "gdrive-distant-association"
    
    # Construct the target path (e.g., gdrive-distant-association:ProjectResults)
    target_path = f"{remote_name}:{remote_folder_name}"
    
    print(f"Starting transfer: {local_path} --> {target_path}")
    
    try:
        # -P shows progress, --update skips files that are already newer on target
        subprocess.run(
            ["rclone", "copy", local_path, target_path, "-P", "--update"], 
            check=True
        )
        print("Transfer completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during transfer: {e}")