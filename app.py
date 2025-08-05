import os
import gdown
from tensorflow.keras.models import load_model

# Define path and Google Drive file ID
model_path = "best_finetuned_model.h5"
file_id = "1iM4KetgPQM-0vIw2raiGOh_Ij9AZCVr61iM4KetgPQM-0vIw2raiGOh_Ij9AZCVr6"  # <-- Your actual ID here
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

# Download the model if not already present
if not os.path.exists(model_path):
    gdown.download(gdrive_url, model_path, quiet=False)

# Load the model
model = load_model(model_path)
