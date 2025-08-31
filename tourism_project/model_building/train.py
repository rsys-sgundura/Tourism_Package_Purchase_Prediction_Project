# ------------------------------------------------------
# Tourism Package Purchase Prediction - Model Training
# ------------------------------------------------------
# This script:
#  1. Loads preprocessed train/test data from Hugging Face Datasets repo.
#  2. Defines numerical and categorical features.
#  3. Builds a preprocessing pipeline (scaling + one-hot encoding).
#  4. Trains an XGBoost classifier with GridSearch hyperparameter tuning.
#  5. Logs all experiments, parameters, and metrics in MLflow.
#  6. Saves the best trained model locally and uploads it to Hugging Face Hub.
# ------------------------------------------------------

# ---------------------------
# Import required libraries
# ---------------------------
# Data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model training and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Saving/loading models
import joblib

# File handling
import os

# Hugging Face Hub (for dataset/model storage)
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow for experiment tracking
import mlflow


# ---------------------------
# 1. MLflow Setup
# ---------------------------
mlflow.set_tracking_uri("http://localhost:5000")      # Tracking server (change if using ngrok/public tunnel)
mlflow.set_experiment("tourism-training-experiment")  # Create or reuse experiment folder

# Hugging Face Hub client (uses HF_TOKEN from environment)
api = HfApi()


# ---------------------------
# 2. Load Preprocessed Data
# ---------------------------
# Load previously uploaded splits from Hugging Face Dataset repo
Xtrain_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/Xtest.csv"
ytrain_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/ytrain.csv"
ytest_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

print("Train/test data successfully loaded from Hugging Face Hub.")


# ---------------------------
# 3. Define Features
# ---------------------------
# Numerical columns → will be scaled (StandardScaler)
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'PitchSatisfactionScore',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
]

# Categorical columns → will be one-hot encoded (OneHotEncoder)
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation',
]


# ---------------------------
# 4. Handle Class Imbalance
# ---------------------------
# Calculate scale_pos_weight = (negative_class / positive_class)
# This tells XGBoost to pay more attention to minority class (customers who purchased)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Calculated class imbalance ratio (scale_pos_weight): {class_weight}")


# ---------------------------
# 5. Preprocessing Pipeline
# ---------------------------
# - StandardScaler applied to numeric features
# - OneHotEncoder applied to categorical features
# - Both combined with ColumnTransformer
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)


# ---------------------------
# 6. Define Model & Hyperparameters
# ---------------------------
# Base XGBoost classifier with imbalance handling
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Grid search parameters (search across multiple values for tuning)
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],   # number of trees
    'xgbclassifier__max_depth': [2, 3, 4],                    # depth of trees
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],       # features per tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],      # features per split level
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],        # learning rate (step size)
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],             # L2 regularization
}

# Combine preprocessing + model into one pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)


# ---------------------------
# 7. Train & Tune Model
# ---------------------------
with mlflow.start_run():  # Begin MLflow run
    print("Starting GridSearchCV for hyperparameter tuning...")

    # GridSearch with 5-fold cross-validation
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log each hyperparameter trial in MLflow (nested runs)
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):  # Log trial results separately
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best hyperparameters in main run
    mlflow.log_params(grid_search.best_params_)
    print("Best parameters found:", grid_search.best_params_)


    # ---------------------------
    # 8. Evaluate Best Model
    # ---------------------------
    best_model = grid_search.best_estimator_

    # Adjust decision threshold (default = 0.5, here tuned to 0.45)
    classification_threshold = 0.45

    # Predictions on train and test sets
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)

    # Generate classification reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log key metrics in MLflow
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })
    print("Training and testing metrics logged in MLflow.")


    # ---------------------------
    # 9. Save Best Model Locally
    # ---------------------------
    model_path = "best_tourism_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")  # also log in MLflow
    print(f"Best model saved locally as {model_path} and logged in MLflow.")


    # ---------------------------
    # 10. Upload Model to Hugging Face Hub
    # ---------------------------
    repo_id = "SudeendraMG/tourism_model"
    repo_type = "model"

    # Create repo if not exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists on Hugging Face.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating it...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created on Hugging Face.")

    # Upload trained model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face Hub at repo: {repo_id}")
