# ------------------------------------------------------
# Tourism Package Purchase Prediction - Data Preparation
# ------------------------------------------------------
# ------------------------------------------------------
# Data Preparation Script for Tourism Dataset
# ------------------------------------------------------
# This script:
#  1. Loads dataset from Hugging Face Hub
#  2. Cleans and preprocesses the data
#  3. Handles missing values and outliers
#  4. Encodes categorical variables
#  5. Normalizes numerical features
#  6. Splits the dataset into train and test sets
#  7. Saves the processed files locally
#  8. Uploads processed files back to Hugging Face Hub
# ------------------------------------------------------

# For data manipulation
import pandas as pd
import sklearn   # (imported for completeness, but not directly used here)
# For file system operations
import os
# For splitting dataset into train/test sets
from sklearn.model_selection import train_test_split
# For Hugging Face Hub authentication and dataset uploads
from huggingface_hub import login, HfApi

# ------------------------------------------------------
# 1. Setup Hugging Face Hub API client
# ------------------------------------------------------
# Uses HF_TOKEN environment variable for authentication
api = HfApi(token=os.getenv("HF_TOKEN"))

# Path to dataset stored on Hugging Face Hub
DATASET_PATH = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/tourism.csv"

# Load dataset into a DataFrame
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ------------------------------------------------------
# Target and Feature Definition
# ------------------------------------------------------
target = 'ProdTaken'  # Target: 1 if customer purchased package, 0 otherwise

# Drop unique identifier (not useful for modeling)
tourism_dataset.drop(columns=['CustomerID'], inplace=True)

# Define numerical features
numeric_features = [
    'Age',                      # Age of the customer
    'CityTier',                 # Tier of the city (1 > 2 > 3)
    'DurationOfPitch',          # Duration of sales pitch (minutes)
    'NumberOfPersonVisiting',   # Total people in the trip
    'NumberOfFollowups',        # Number of follow-ups made
    'PreferredPropertyStar',    # Preferred hotel star rating
    'NumberOfTrips',            # Average annual trips
    'Passport',                 # 1 if has passport, 0 otherwise
    'OwnCar',                   # 1 if owns a car, 0 otherwise
    'PitchSatisfactionScore',   # Customer's satisfaction score for the pitch
    'NumberOfChildrenVisiting', # Number of children <5 accompanying
    'MonthlyIncome',            # Customer's monthly income
]

# Define categorical features
categorical_features = [
    'TypeofContact',   # Method of contact (Company Invited / Self Inquiry)
    'Occupation',      # Occupation type (Salaried, Freelancer, etc.)
    'Gender',          # Gender (Male, Female, "Fe Male")
    'ProductPitched',  # Type of tourism product pitched
    'MaritalStatus',   # Marital status (Single, Married, Divorced)
    'Designation',     # Job designation (Executive, Manager, VP, etc.)
]

# ------------------------------------------------------
# 2. Data Cleaning
# ------------------------------------------------------
# Fix inconsistent values in Gender column
tourism_dataset['Gender'] = tourism_dataset['Gender'].replace({'Fe Male': 'Female'})

# ------------------------------------------------------
# 3, 4, 5. Prepare predictor (X) and target (y)
# ------------------------------------------------------
X = tourism_dataset[numeric_features + categorical_features]  # 18 predictor columns
y = tourism_dataset[target]                                   # Target column

# ------------------------------------------------------
# 6. Train-Test Split
# ------------------------------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,     # 20% data kept for testing
    random_state=42    # Ensures reproducibility
)

# ------------------------------------------------------
# 7. Save splits locally
# ------------------------------------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# ------------------------------------------------------
# 8. Upload train-test splits to Hugging Face Hub
# ------------------------------------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # Upload with filename only
        repo_id="SudeendraMG/tourism-package-purchase-prediction",
        repo_type="dataset",
    )

print("Train-test splits uploaded to Hugging Face Hub successfully.")
