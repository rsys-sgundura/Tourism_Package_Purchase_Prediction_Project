# ------------------------------------------------------
# Register dataset with Hugging Face Hub
# ------------------------------------------------------

# Hugging Face Hub utilities
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# ------------------------------------------------------
# Define repository details
# ------------------------------------------------------
repo_id = "SudeendraMG/tourism-package-purchase-prediction"  # unique repo name on Hugging Face
repo_type = "dataset"  # since we are uploading data, repo type is "dataset"

# ------------------------------------------------------
# Initialize API client with authentication token
# (token should be set in environment variable HF_TOKEN)
# ------------------------------------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

# ------------------------------------------------------
# Step 1: Check if the dataset repository already exists
# ------------------------------------------------------
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    # If repo does not exist, create a new one
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# ------------------------------------------------------
# Step 2: Upload dataset folder contents to Hugging Face Hub
# ------------------------------------------------------
# This will upload all files inside tourism_project/data/ (e.g., tourism.csv)
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Dataset uploaded successfully to Hugging Face Hub.")
