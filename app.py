
import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import base64


def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    
    return href

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    
    chest_pain_options = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    resting_ecg_options = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    st_slope_options = ["Upsloping", "Flat", "Downsloping"]

    age = st.number_input("Age (years)", min_value=0, max_value=150, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", chest_pain_options)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", resting_ecg_options)
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202, value=150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=0.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", st_slope_options)
    
    
    sex_num = 1 if sex == "Male" else 0  
    chest_pain_num = chest_pain_options.index(chest_pain)
    fasting_bs_num = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg_num = resting_ecg_options.index(resting_ecg)
    exercise_angina_num = 1 if exercise_angina == "Yes" else 0
    st_slope_num = st_slope_options.index(st_slope)

    # Create a DataFrame with user inputs matching training feature names exactly
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_num],
        'ChestPainType': [chest_pain_num],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_num],
        'RestingECG': [resting_ecg_num],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_num],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_num]
    })

algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'GridRandom']
modelnames = ['tree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl', 'gridrf.pkl']

def predict_heart_disease(data):
    predictions_list = []
    for modelname in modelnames:
        with open(modelname, 'rb') as f:
            model = pickle.load(f)
        prediction = model.predict(data)
        predictions_list.append(prediction)
    return predictions_list

# Create a submit button to make predictions
if st.button("Submit"):
    st.subheader('Results....')
    st.markdown('----------------------------')
    
    try:
        result = predict_heart_disease(input_data)
        
        for i in range(len(result)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("🟢 No heart disease detected.")
            else:
                st.write("🔴 Heart disease detected.")
            st.markdown('----------------------------')
            
    except EOFError:
        st.error("🚨 Error loading your trained models. One or more `.pkl` files in your directory are empty or corrupted (0 KB). Please re-run your training notebook cells to save them properly.")
        
        
with tab2:
    st.title("Upload CSV File")
    
    st.subheader('Instructions to note before uploading the file:')
    st.info("""
    1. No NaN values allowed.
    2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
    3. Check the spellings of the feature names.
    4. Feature values conventions: \n
    - Age: age of the patient [years] \n
    - Sex: sex of the patient [0: Male, 1: Female] \n
    - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2:Asymptomatic]
    - RestingBP: resting blood pressure [mm Hg] \n
    - Cholesterol: serum cholesterol [mm/dl] \n
    - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise] \n
    - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversion]
    - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202] \n
    - ExerciseAngina: exercise-induced angina [1: Yes, 0: No] \n
    - Oldpeak: oldpeak = ST [Numeric value measured in depression] \n
    - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping] \n
    """)
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # 1. Read the uploaded CSV file into a DataFrame
        bulk_input_data = pd.read_csv(uploaded_file)
        
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        # Remove extra whitespace from column names if present
        bulk_input_data.columns = bulk_input_data.columns.str.strip()

        # 2. Validate that all required columns are present
        if set(expected_columns).issubset(bulk_input_data.columns):
            try:
                # Create a temporary copy to convert text to numbers safely
                processed_data = bulk_input_data[expected_columns].copy()

                # Clean any string whitespaces inside the data rows
                for col in processed_data.columns:
                    if processed_data[col].dtype == 'object':
                        processed_data[col] = processed_data[col].astype(str).str.strip()

                # 3. DATA CLEANING: Convert text categories to numeric representations
                
                # Convert Sex: 'M'/'Male' -> 0, 'F'/'Female' -> 1
                if processed_data['Sex'].dtype == 'object':
                    processed_data['Sex'] = processed_data['Sex'].map({'M': 0, 'Male': 0, 'F': 1, 'Female': 1})
                
                # Convert ExerciseAngina: 'No'/'N' -> 0, 'Yes'/'Y' -> 1
                if processed_data['ExerciseAngina'].dtype == 'object':
                    processed_data['ExerciseAngina'] = processed_data['ExerciseAngina'].map({'No': 0, 'N': 0, 'Yes': 1, 'Y': 1})

                # Convert ChestPainType text strings to numeric values
                if processed_data['ChestPainType'].dtype == 'object':
                    cp_map = {"Typical Angina": 3, "Atypical Angina": 0, "Non-Anginal Pain": 1, "Asymptomatic": 2, "TA": 3, "ATA": 0, "NAP": 1, "ASY": 2}
                    processed_data['ChestPainType'] = processed_data['ChestPainType'].map(cp_map)

                # Convert ST_Slope text strings to numeric values
                if processed_data['ST_Slope'].dtype == 'object':
                    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2, "Up": 0, "Flat": 1, "Down": 2}
                    processed_data['ST_Slope'] = processed_data['ST_Slope'].map(slope_map)

                # Convert remaining data blocks to numeric objects just in case
                for col in expected_columns:
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

                # Fallback: fill any unmapped or missing items with 0
                processed_data = processed_data.fillna(0)

                # 4. Load the prediction model
                with open('LogisticRegression.pkl', 'rb') as f:
                    model = pickle.load(f)
                
                # 5. Predict using a clean float matrix array (CRITICAL FIX FOR VALUE ERROR)
                features = processed_data.values.astype(float)
                bulk_input_data['Prediction LR'] = model.predict(features)
                
                # 6. Save the outputs locally
                bulk_input_data.to_csv('PredictedHeartLR.csv', index=False)

                # Display the data table with predictions
                st.subheader("Predictions:")
                st.write(bulk_input_data)

                # Provide the HTML download button link
                st.markdown(get_binary_file_downloader_html(bulk_input_data), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An unexpected error occurred during processing: {e}")
        else:
            st.warning("Please make sure the uploaded CSV file has the correct columns.")
            
            


with tab3:
    import plotly.express as px
    data = {'Decision Trees':80.97,'Logistic Regression':85.86,'Random Forest':84.23,'Support Vector Machine':84.22,'GridRF':89.75}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
    fig = px.bar(df,y='Accuracies',x='Models')
    st.plotly_chart(fig)
    