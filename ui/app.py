import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="Heart Disease Detection System",
    page_icon="♥️",
    layout="wide"
)


# Load the model
@st.cache_resource
def load_model():
    return joblib.load('ml_model/heart_disease_dt_model.joblib')


# Load the data for visualization
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data.csv")


# Initialize
model = load_model()
df = load_data()


# Function to preprocess user input
def preprocess_input(input_data):
    # Create a DataFrame with the same columns as our training data
    # We'll need to match this to your model's expected features
    X_columns = df.drop('target', axis=1).columns

    # Create empty DataFrame with matching columns
    user_df = pd.DataFrame(columns=X_columns)

    # Add the user input data
    user_df.loc[0] = input_data

    return user_df


# App title
st.title("♥️ Heart Disease Detection System")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Data Visualization", "About"])

# Home page
if page == "Home":
    st.header("Welcome to the Heart Disease Detection System")
    
    st.image("ui/heart-disease-740.jpg",
            caption="Stay heart-healthy with regular checkups and a healthy lifestyle.")
           
    st.markdown("""
    This application helps predict the risk of heart disease based on various health indicators.

    ### Features:
    - **Prediction**: Enter your health data and receive a heart disease risk assessment
    - **Visualization**: Explore patterns and correlations in heart disease data
    - **Decision Tree Model**: Utilizes a machine learning model with accuracy of approximately 98.5%

    ### How to use:
    1. Navigate to the **Prediction** page
    2. Enter your health information
    3. Click the "Predict" button to see your risk assessment

    ### Data Visualization:
    Explore various charts and statistics in the **Data Visualization** page
    """)

# Prediction page
elif page == "Prediction":
    st.header("Heart Disease Risk Prediction")
    st.markdown("Enter your health information below to get a risk assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 29, 77, 54)
        sex = st.radio("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 126, 564, 200)

    with col2:
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results",
                               ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.slider("Maximum Heart Rate", 71, 202, 150)
        exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, 0.1)

    with col3:
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Convert categorical inputs to numerical
    sex_encoded = 1 if sex == "Male" else 0

    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    cp_encoded = cp_mapping[cp]

    fbs_encoded = 1 if fbs == "Yes" else 0

    restecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    restecg_encoded = restecg_mapping[restecg]

    exang_encoded = 1 if exang == "Yes" else 0

    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope_encoded = slope_mapping[slope]

    thal_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal_encoded = thal_mapping[thal]

    # Normalize the numerical features (as done in your preprocessing step)
    # These min-max ranges should match your training data
    age_norm = (age - 29) / (77 - 29)
    trestbps_norm = (trestbps - 94) / (200 - 94)
    chol_norm = (chol - 126) / (564 - 126)
    thalach_norm = (thalach - 71) / (202 - 71)
    oldpeak_norm = oldpeak / 6.2
    ca_norm = ca / 4

    # Create input data in the same format as your model expects
    input_data = [age_norm, sex_encoded, cp_encoded, trestbps_norm, chol_norm,
                  fbs_encoded, restecg_encoded, thalach_norm, exang_encoded,
                  oldpeak_norm, slope_encoded, ca_norm, thal_encoded]

    # Make sure the input data matches your model's feature expectations
    # You might need to adjust this based on how one-hot encoding was applied

    # Predict button
    if st.button("Predict"):
        try:
            # Preprocess the input data to match the model's expectations
            user_df = preprocess_input(input_data)

            # Get prediction
            prediction = model.predict(user_df)
            probability = model.predict_proba(user_df)

            # Display result
            st.subheader("Prediction Results")

            if prediction[0] == 1:
                st.error("⚠️ High Risk of Heart Disease")
                risk_percentage = probability[0][1] * 100
                st.markdown(f"The model predicts a **{risk_percentage:.2f}%** probability of heart disease.")
            else:
                st.success("✅ Low Risk of Heart Disease")
                safe_percentage = probability[0][0] * 100
                st.markdown(f"The model predicts a **{safe_percentage:.2f}%** probability of no heart disease.")

            # Feature importance for this prediction
            st.subheader("Feature Contribution")
            feature_names = user_df.columns

            # Create a bar chart of the feature values
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_names, user_df.values[0])
            ax.set_xlabel('Normalized Value')
            ax.set_title('Your Health Indicators')
            st.pyplot(fig)

            # Recommendations based on prediction
            st.subheader("Recommendations")
            if prediction[0] == 1:
                st.markdown("""
                - Consult with a healthcare professional as soon as possible
                - Monitor your blood pressure and cholesterol regularly
                - Consider lifestyle changes including diet and exercise
                - Follow up with regular cardiac checkups
                """)
            else:
                st.markdown("""
                - Continue maintaining a healthy lifestyle
                - Regular checkups are still recommended
                - Monitor any changes in your health
                - Stay active and maintain a balanced diet
                """)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Data Visualization page
elif page == "Data Visualization":
    st.header("Data Visualization Dashboard")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlation", "Target Analysis", "Feature Importance"])

    with tab1:
        st.subheader("Feature Distributions")

        # Select features to display
        selected_features = st.multiselect(
            "Select features to visualize",
            options=df.columns.tolist(),
            default=["age", "chol", "thalach", "oldpeak"]
        )

        if selected_features:
            # Create histograms
            fig, axes = plt.subplots(len(selected_features), 1, figsize=(10, len(selected_features) * 3))

            if len(selected_features) == 1:
                axes = [axes]

            for i, feature in enumerate(selected_features):
                sns.histplot(df[feature], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f"Distribution of {feature}")

            plt.tight_layout()
            st.pyplot(fig)

            # Boxplots
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df[selected_features], palette="Set2", ax=ax)
            plt.xticks(rotation=45)
            plt.title("Boxplot of Selected Features")
            st.pyplot(fig)
        else:
            st.info("Please select at least one feature to visualize")

    with tab2:
        st.subheader("Correlation Analysis")

        # Correlation heatmap
        corr_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.title("Feature Correlation Heatmap")
        st.pyplot(fig)

        # Correlation with target
        target_corr = corr_matrix['target'].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        target_corr.drop('target').plot(kind='bar', ax=ax)
        plt.title("Correlation with Heart Disease")
        plt.ylabel("Correlation Coefficient")
        st.pyplot(fig)

    with tab3:
        st.subheader("Target Analysis")

        # Distribution of target variable
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='target', data=df, palette=['skyblue', 'salmon'], ax=ax)
        plt.title("Heart Disease Distribution")
        plt.xlabel("Heart Disease (0=No, 1=Yes)")
        plt.ylabel("Count")

        # Add count and percentage labels
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            percentage = round((height / total) * 100, 1)
            ax.text(p.get_x() + p.get_width() / 2., height + 5, f'{int(height)} ({percentage}%)',
                    ha="center", fontsize=10)

        st.pyplot(fig)

        # Feature comparison by target
        st.subheader("Feature Comparison by Heart Disease Status")

        compare_feature = st.selectbox(
            "Select feature to compare",
            options=[col for col in df.columns if col != 'target'],
            index=0
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='target', y=compare_feature, data=df, palette=['skyblue', 'salmon'], ax=ax)
        plt.title(f"{compare_feature} by Heart Disease Status")
        plt.xlabel("Heart Disease (0=No, 1=Yes)")
        st.pyplot(fig)

    with tab4:
        st.subheader("Feature Importance")

        from sklearn.ensemble import ExtraTreesClassifier

        # Train a feature importance model
        X = df.drop(columns=['target'])
        y = df['target']

        model = ExtraTreesClassifier(random_state=42)
        model.fit(X, y)

        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Get feature names from X
        feature_names = X.columns
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importances = feature_importances.sort_values('importance', ascending=False)

        # Plot feature importances
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances, ax=ax)
        ax.set_title('Feature Importance for Heart Disease Prediction')
        st.pyplot(fig)

# About page
elif page == "About":
    st.header("About This Project")

    st.markdown("""
    ### Heart Disease Detection System

    This application was developed as part of a project to combine rule-based expert systems with machine learning for heart disease risk assessment.

    #### Key Components:
    - **Data Processing**: Cleaning, normalization, and feature engineering
    - **Decision Tree Model**: Trained with optimized hyperparameters (98.5% accuracy)
    - **Interactive UI**: User-friendly interface for risk prediction
    - **Visualization Dashboard**: Data exploration and analysis tools

    #### Dataset Information:
    The system uses a heart disease dataset with the following features:
    - Age, sex, chest pain type
    - Blood pressure, cholesterol, blood sugar
    - ECG results, maximum heart rate
    - Exercise-induced angina, ST depression
    - Slope of peak exercise ST segment
    - Number of major vessels colored by fluoroscopy
    - Thalassemia

    #### References:
    - UCI Heart Disease Dataset
    - Python libraries: scikit-learn, pandas, numpy, matplotlib, seaborn, streamlit
    """)

    st.markdown("---")
    st.markdown("Developed as part of the Heart Disease Detection Project for Expert Systems")

# Add footer
st.markdown("---")
st.markdown("© 2025 Heart Disease Detection System | Created by Yakoot, Malak and Omar")