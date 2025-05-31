import streamlit as st
import numpy as np 
import pandas as pd 
import pickle
import base64


#Function to create a download Link for a DataFrame as a CSV file
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Prediction CSV</a>'


st.title("Heart Disease Predictor")
tab1,tab2,tab3 = st.tabs(['Predict','Bulk Predict','Model Information'])

# Example placeholder for models
def load_models():
    model_files = ['Tree.pkl', 'LogisticRegression.pkl', 'RandomForest.pkl', 'SVM.pkl']
    return [pickle.load(open(model, 'rb')) for model in model_files]

# Function to make predictions
def predict_heart_disease(input_df):
    models = load_models()
    results = []
    for model in models:
        prediction = model.predict(input_df)
        results.append(prediction)
    return results

# Tab/section for inputs
with st.container():  # Use st.container() or tabs if needed
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numeric
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Prepare DataFrame
    inputs_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
    })

    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']

    # Submit button
    if st.button("Submit"):
        st.subheader('Results...')
        st.markdown('------------------------')

        try:
            result = predict_heart_disease(inputs_data)
            for i in range(len(result)):
                st.subheader(algonames[i])
                if result[i][0] == 0:
                    st.write("No heart disease detected.")
                else:
                    st.write("Heart disease detected.")
                st.markdown('-------------------------')
        except Exception as e:
            st.error(f"Error making prediction: {e}")

with tab2:
    st.title("Upload CSV File")

    st.subheader('Instructions to note before uploading the file:')
    st.info("""
    1. No NaN values allowed.
    2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n 
    3. Check the spellings of the feature names.
    4.Feature values convention: \n
        - Age: age of the patient [years] \n
        - Sex: sex of the patient [0: Male, 1: Female] \n
        - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic] \n
        - RestingBP: resting blood pressure [mm Hg] \n
        - Cholesterol: serum cholesterol [mm/dl] \n
        - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise] \n
        - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality, 2:"Left Ventricular Hypertrophy"]) \n
        - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202] \n
        - ExerciseAngina: Ingina: exercise-induced angina [1: Yes, 0: No] \n
        - Oldpeak: oldpeak = ST [Numeric value measured in depression] \n
        - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping] \n



    """)
    # Create a file uploader in the sidebar
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open('LogisticRegression.pkl','rb'))

        # Ensure that the input DataFrame matches the expected columns and format
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        input_data['Prediction LR'] = ''

        for i in range(len(input_data)):
            arr = input_data.iloc[i,:-1].values
            input_data['Prediction LR'][i] = model.predict([arr])[0]
        input_data.to_csv('PredictedHeartLR.csv')

        # Display the predictions
        st.subheader("Predictions:")
        st.write(input_data)

        # Create a button to download the updated CSV file
        st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        st.warning("Pleasse make sure the uploaded CSV file has the correct columns. ")
    else:
        st.info("Upload a CSV file to get prediction.")


with tab3:
    import plotly.express as px
    data = {'Decision Trees': 89.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine': 84.22}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
    fig = px.bar(df,y='Accuracies',x='Models')
    st.plotly_chart(fig)