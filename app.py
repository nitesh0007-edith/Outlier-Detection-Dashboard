import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st
from datetime import datetime
from PIL import Image
import base64
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Set up background image
def set_background_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode()
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_image}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # Calculate BMI if both columns are available
    if 'WEIGHT_AT_START_OF_REGIMEN' in data.columns and 'HEIGHT_AT_START_OF_REGIMEN' in data.columns:
        data['HEIGHT_AT_START_OF_REGIMEN'] = data['HEIGHT_AT_START_OF_REGIMEN'] / 100  # Convert height to meters
        data['BMI'] = data['WEIGHT_AT_START_OF_REGIMEN'] / (data['HEIGHT_AT_START_OF_REGIMEN'] ** 2)
        st.write("BMI column calculated and added to the dataset.")
    else:
        st.warning("Both 'WEIGHT_AT_START_OF_REGIMEN' and 'HEIGHT_AT_START_OF_REGIMEN' columns are required to calculate BMI.")
    
    return data

# Function to create and download data profiling report
def create_profiling_report(data):
    profile = ProfileReport(data, title="Data Profiling Report", explorative=True)
    st_profile_report(profile)
    
    # Save report to HTML and provide download link
    profile.to_file("data_profiling_report.html")
    with open("data_profiling_report.html", "rb") as file:
        st.download_button(
            label="Download Data Profiling Report",
            data=file,
            file_name="data_profiling_report.html",
            mime="text/html"
        )

# Outlier detection functions
def detect_outliers_zscore(data, column, threshold=3):
    col_data = data[column].dropna()
    z_scores = np.abs(stats.zscore(col_data))
    outliers = col_data[z_scores > threshold]
    return data.loc[outliers.index]

def detect_outliers_iqr(data, column):
    col_data = data[column].dropna()
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))]
    return data.loc[outliers.index]

# Date mismatch detection and plotting function with error handling
def detect_date_mismatch(data, start_column, decision_column):
    data[start_column] = pd.to_datetime(data[start_column], errors='coerce')
    data[decision_column] = pd.to_datetime(data[decision_column], errors='coerce')
    
    mismatches = data[data[start_column] < data[decision_column]]
    return mismatches

def plot_date_mismatches(data, start_column, decision_column):
    mismatches = detect_date_mismatch(data, start_column, decision_column)
    mismatches = mismatches.dropna(subset=[start_column, decision_column])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(mismatches[decision_column], mismatches[start_column], color="red", label="Mismatches")
    plt.plot(data[decision_column], data[decision_column], linestyle='--', color='blue', label="Expected")
    plt.title("Date Mismatches (Decision to Treat vs. Start Date of Regimen)")
    plt.xlabel("Date of Decision to Treat")
    plt.ylabel("Start Date of Regimen")
    plt.legend()
    st.pyplot(plt)
    plt.close()


# Visualization functions
def plot_boxplot(data, column, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column].dropna(), color="purple", saturation=0.6)
    plt.title(title)
    plt.xlabel(column)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)
    plt.close()

def plot_distribution(data, column, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column].dropna(), kde=True, color="teal", edgecolor="black")
    mean, median, mode, std_dev = data[column].mean(), data[column].median(), data[column].mode().iloc[0], data[column].std()
    plt.axvline(mean, color="red", linestyle="--", linewidth=1, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color="blue", linestyle="--", linewidth=1, label=f'Median: {median:.2f}')
    plt.axvline(mode, color="green", linestyle="--", linewidth=1, label=f'Mode: {mode:.2f}')
    plt.title(title)
    plt.xlabel(column)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)
    plt.close()

# Imputation functions with explanations
def impute_outliers_median(data, column):
    median = data[column].median()
    st.write(f"Median Imputation: Using median value {median} as it reduces the influence of extreme outliers and is a robust measure for skewed data.")
    data[column] = np.where((data[column] > data[column].quantile(0.95)) | 
                            (data[column] < data[column].quantile(0.05)), median, data[column])
    return data

def impute_outliers_mean(data, column):
    mean = data[column].mean()
    st.write(f"Mean Imputation: Using mean value {mean:.2f}, suitable for normally distributed data without extreme outliers.")
    data[column] = np.where((data[column] > data[column].quantile(0.95)) | 
                            (data[column] < data[column].quantile(0.05)), mean, data[column])
    return data

def impute_outliers_mode(data, column):
    mode = data[column].mode()[0]
    st.write(f"Mode Imputation: Using mode value {mode}, often appropriate for categorical or discrete variables.")
    data[column] = np.where((data[column] > data[column].quantile(0.95)) | 
                            (data[column] < data[column].quantile(0.05)), mode, data[column])
    return data

def impute_locf(data, column):
    st.write("LOCF Imputation: Chosen for longitudinal data, carrying forward the last observation for missing or outlying values, which maintains temporal consistency.")
    data[column] = data[column].fillna(method='ffill')
    return data

# Streamlit dashboard
def dashboard():
    st.title("Outlier Detection Dashboard")
    set_background_image("/Users/niteshranjansingh/Downloads/Challenge_7/gradient-network-connection-background_23-2148879890.avif")  # Update with your image path
    
    st.markdown("""
    ### Welcome to the Outlier Detection Dashboard!

    This tool is designed to simplify the process of identifying and handling outliers in healthcare datasets, particularly for projects that rely on real-world evidence data. Here, healthcare data analysts and professionals can quickly visualize and address outliers that may impact analysis accuracy. 

    With this dashboard, you can:
    - **Upload and explore** patient data with ease.
    - **Identify outliers** in key patient variables such as age, height, weight, BMI, and treatment dates, using robust methods like Z-score and IQR.
    - **Visualize** these outliers interactively through clear graphs and charts.
    - **Resolve inconsistencies** by imputing or adjusting outlier values using multiple imputation methods, including median, mean, mode, and last observation carried forward (LOCF).
    - **Check for date mismatches** in treatment timelines to ensure data consistency in longitudinal patient records.

    This dashboard was created to streamline healthcare data analysis, ensuring fast, insightful, and reliable data cleaning and exploration. Let’s get started with your dataset!
    """)
    
    file_path = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    
    if file_path:
        data = load_data(file_path)
        st.write("Dataset Preview:", data.head())
        
        if st.button("Generate and Download Data Profiling Report"):
            create_profiling_report(data)
        
        main_method = st.selectbox("Select the main task", ["Outlier Detection", "Outlier Imputation", "Date Mismatch"])
        
        numeric_columns = [col for col in data.select_dtypes(include=[np.number]).columns if col not in ["PATIENTID", "LINKNUMBER"]]
        
        if main_method == "Outlier Detection":
            variable = st.selectbox("Choose a variable for Outlier Detection", numeric_columns)
            detection_method = st.radio("Choose a detection method", ["Z-score", "IQR"])
            
            if st.button("Identify Outliers"):
                if detection_method == "Z-score":
                    outliers = detect_outliers_zscore(data, variable)
                    st.write("Detected Outliers:", outliers)
                    plot_boxplot(data, variable, f"{variable} - Outliers by Z-score")
                    
                elif detection_method == "IQR":
                    outliers = detect_outliers_iqr(data, variable)
                    st.write("Detected Outliers:", outliers)
                    plot_boxplot(data, variable, f"{variable} - Outliers by IQR")
        
        elif main_method == "Outlier Imputation":
            variable = st.selectbox("Choose a variable for Outlier Imputation", numeric_columns)
            imputation_method = st.radio("Choose an imputation method", ["Median Imputation", "Mean Imputation", "Mode Imputation", "LOCF"])
            
            if st.button("Impute Outliers"):
                if imputation_method == "Median Imputation":
                    imputed_data = impute_outliers_median(data, variable)
                elif imputation_method == "Mean Imputation":
                    imputed_data = impute_outliers_mean(data, variable)
                elif imputation_method == "Mode Imputation":
                    imputed_data = impute_outliers_mode(data, variable)
                elif imputation_method == "LOCF":
                    imputed_data = impute_locf(data, variable)
                
                st.write("Data with imputed outliers:", imputed_data.head())
                plot_distribution(imputed_data, variable, f"{variable} - Distribution After Imputation")
                
                # Provide option to download imputed dataset
                csv_data = imputed_data.to_csv(index=False)
                st.download_button(
                    label="Download Imputed Dataset",
                    data=csv_data,
                    file_name="imputed_data.csv",
                    mime="text/csv"
                )
        
        elif main_method == "Date Mismatch":
            start_column = st.selectbox("Choose the start date column", data.columns)
            decision_column = st.selectbox("Choose the decision date column", data.columns)
            
            if st.button("Detect Date Mismatches"):
                mismatches = detect_date_mismatch(data, start_column, decision_column)
                st.write("Date Mismatches:", mismatches)
                plot_date_mismatches(data, start_column, decision_column)

if __name__ == "__main__":
    dashboard()
