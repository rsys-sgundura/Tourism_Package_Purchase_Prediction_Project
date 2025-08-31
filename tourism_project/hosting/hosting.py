# --------------------------------------------------------
# Hosting Script: Upload Deployment Files to Hugging Face Space
# --------------------------------------------------------
# This script automates pushing the local deployment files 
# (e.g., Dockerfile, app.py, requirements.txt) to a Hugging Face 
# "Space" repository so that the app can be hosted and run directly.
# --------------------------------------------------------

from huggingface_hub import HfApi   # Import Hugging Face API to interact with HF Hub
import os                           # For accessing environment variables like HF_TOKEN

# Initialize the Hugging Face API with authentication using your HF token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the entire "deployment" folder to the Hugging Face Space repository
api.upload_folder(
    folder_path="tourism_project/deployment",        # Local folder that contains all deployment files
    repo_id="SudeendraMG/Tourism-Package-Purchase-Prediction",  # Hugging Face Space repo name
    repo_type="space",                               # Type of repo (dataset, model, or space → here 'space')
    path_in_repo="",                                 # (Optional) Upload location inside repo; "" = root
)

# --------------------------------------------------------
# After this script runs successfully:
# - The files in tourism_project/deployment will be available
#   inside the Hugging Face Space repository.
# - Hugging Face will automatically build and host the app.
# --------------------------------------------------------
