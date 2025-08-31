# --------------------------------------------
# STEP 1: Import Required Libraries
# --------------------------------------------
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download   # For downloading model from Hugging Face Hub
import joblib                                 # For loading the trained ML model

# --------------------------------------------
# STEP 2: Load the Trained Model
# --------------------------------------------
# Download the saved XGBoost model file from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="SudeendraMG/tourism_model", 
    filename="best_tourism_prediction_model_v1.joblib"
)

# Load the model into memory
model = joblib.load(model_path)

# --------------------------------------------
# STEP 3: Build Streamlit UI
# --------------------------------------------
st.title("Tourism Package Purchase Prediction App")  # Application title

# Short description about the app
st.write("""
The Tourism Package Purchase Prediction App is a tool used by the travel company named **"Visit with Us"**.  
  
The App helps to predict whether a customer would purchase the newly introduced **Wellness Tourism Package**.

""")

st.write("Please enter the customer details to check whether he/she will likely take up for the tourism package.")

# --------------------------------------------
# STEP 4: Collect User Input via Streamlit Widgets
# --------------------------------------------
# Numerical inputs
Age = st.number_input("Age (customer's age in years)", min_value=18.0, max_value=110.0, value=18.0, step=1.0)
CityTier = st.selectbox(
    "City category (based on development, population, and living standards)",
    ["Tier 1", "Tier 2", "Tier 3"]
)
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer on the trip", min_value=0, max_value=30, value=0, step=1)
PreferredPropertyStar = st.number_input("Preferred hotel rating by the customer", min_value=1.0, max_value=7.0, value=3.0, step=1.0)
NumberOfTrips = st.number_input("Average number of trips the customer takes annually", min_value=0.0, value=1.0, step=1.0)
Passport = st.selectbox("Does the customer hold a valid passport?", ["Yes", "No"])
OwnCar = st.selectbox("Does the customer own a car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer", min_value=0.0, value=0.0, step=1.0)
MonthlyIncome = st.number_input("Gross monthly income of the customer", min_value=0.0, value=5000.0)
PitchSatisfactionScore = st.number_input("Score indicating satisfaction with the sales pitch (1–5)", min_value=1, max_value=5, value=1, step=1)
NumberOfFollowups = st.number_input("Total number of follow-ups after sales pitch", min_value=0.0, value=1.0, step=1.0)
DurationOfPitch = st.number_input("Duration of the sales pitch (in minutes)", min_value=1.0, value=1.0, step=1.0)

# Categorical inputs
TypeofContact = st.selectbox("How was the customer contacted?", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Customer's occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender of the customer", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital status of the customer", ["Married", "Divorced", "Unmarried", "Single"])
Designation = st.selectbox("Designation in current organization", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
ProductPitched = st.selectbox("Type of product pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

# --------------------------------------------
# STEP 5: Convert Input Data to Model Format
# --------------------------------------------
# Mapping city tier labels into numeric values (same as training dataset)
citytier_mapping = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3}

# Prepare a single-row DataFrame for model input
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': citytier_mapping[CityTier],
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched
}])

# --------------------------------------------
# STEP 6: Make Prediction
# --------------------------------------------
# Set decision threshold (adjustable for precision-recall tradeoff)
classification_threshold = 0.45

# Predict when button is clicked
if st.button("Predict"):
    # Get probability of opting for package
    prediction_proba = model.predict_proba(input_data)[0, 1]
    
    # Convert probability into binary outcome using threshold
    prediction = (prediction_proba >= classification_threshold).astype(int)
    
    # Final result message
    result = "Take-up For Tourism Package :)" if prediction == 1 else "NOT take-up Tourism Package :("
    
    # Display result
    st.subheader("Prediction Result")
    st.write(f"Based on the above user info, the customer may: **{result}**")
