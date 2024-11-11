import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
@st.cache
def load_datasets():
    data1 = pd.read_csv('data/Student_performance_data.csv')
    data2 = pd.read_csv('data/gpa_study_hours.csv')
    data3 = pd.read_csv('data/FirstYearGPA.csv')
    return data1, data2, data3

data1, data2, data3 = load_datasets()

# Sidebar for dataset selection
st.sidebar.header('Dataset Selection')
st.sidebar.write("Interested in my other analysis?")
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["Student Performance", 
     "First-Year GPA", 
     "SAT Scores and GPA"])

# Define thesis and research questions
st.title("Exploring Factors That Influence Academic Success")
st.write("""
### Thesis
**"How do study habits, academic background, and demographic factors influence first-year GPA and academic success in college students?"**

This analysis aims to uncover which factors impact GPA, using three datasets. We’ll examine relationships between study hours, demographics, and academic performance to provide actionable insights for student support systems.
""")

# Visualizations and Analysis per Dataset
if dataset_name == "Student Performance":
    st.header("Dataset 1: Student Performance")
    st.write("Exploring how different academic scores and demographics relate to student performance.")

    # Basic Data Overview
    st.subheader("Data Preview")
    st.write(data1.head())

    # Questions to Answer
    st.subheader("Research Questions")
    st.write("""
    1. How does study time, absences, and parental support influence GPA?
    2. Are there performance differences across demographic groups, such as gender or parental education level?
    """)

    # Data Visualization
    st.subheader("Distribution of GPA")
    fig, ax = plt.subplots()
    sns.histplot(data1['GPA'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Study Time vs. GPA")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data1['StudyTimeWeekly'], y=data1['GPA'], hue=data1['Gender'], ax=ax)
    st.pyplot(fig)

elif dataset_name == "First-Year GPA":
    st.header("Dataset 3: First-Year GPA")
    st.write("Examining how demographic factors influence first-year college GPA.")

    # Basic Data Overview
    st.subheader("Data Preview")
    st.write(data3.head())

    # Questions to Answer
    st.subheader("Research Questions")
    st.write("""
    1. How do factors like high school GPA, SAT scores, and first-generation status influence first-year college GPA?
    2. Is there a significant difference in GPA among students from different socioeconomic backgrounds?
    """)

    # Data Visualization
    st.subheader("Distribution of First-Year GPA")
    fig, ax = plt.subplots()
    sns.histplot(data3['GPA'], kde=True, ax=ax)
    st.pyplot(fig)

    # Modeling GPA based on selected features
    st.subheader("Predicting First-Year GPA Based on Demographics")
    selected_features = ['HSGPA', 'SATV', 'SATM', 'Male', 'FirstGen']  # Adjusted to available features
    X = data3[selected_features]
    y = data3['GPA']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    st.write("Mean Squared Error (Random Forest):", mean_squared_error(y_test, y_pred_rf))
    st.write("R2 Score (Random Forest):", r2_score(y_test, y_pred_rf))

elif dataset_name == "SAT Scores and GPA":
    st.header("Dataset: SAT Scores and GPA")
    st.write("Analyzing the correlation between SAT scores and GPA.")
    
    # Display data preview
    st.subheader("Data Preview")
    st.write(data3.head())
    
    # Research Questions
    st.subheader("Research Questions")
    st.write("""
    1. Can SAT scores predict high school GPA?
    2. Can SAT scores predict college GPA?
    3. How strong is the correlation between SAT scores and GPAs?
    """)
    
    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")
    
    # Scatter Plot: SAT Verbal vs High School GPA
    fig, ax = plt.subplots()
    sns.scatterplot(data=data3, x='SATV', y='HSGPA', ax=ax)
    ax.set_title('SAT Verbal Score vs High School GPA')
    st.pyplot(fig)

    # Scatter Plot: SAT Math vs College GPA
    fig, ax = plt.subplots()
    sns.scatterplot(data=data3, x='SATM', y='GPA', ax=ax)
    ax.set_title('SAT Math Score vs College GPA')
    st.pyplot(fig)

    # Correlation Coefficients
    st.subheader("Correlation Coefficients")
    corr_hs = data3['SATV'].corr(data3['HSGPA'])
    corr_college = data3['SATM'].corr(data3['GPA'])
    st.write(f"Correlation between SAT Verbal Score and High School GPA: **{corr_hs:.2f}**")
    st.write(f"Correlation between SAT Math Score and College GPA: **{corr_college:.2f}**")

    # Model Training
    st.subheader("Predictive Modeling")

    # Select GPA to Predict
    gpa_option = st.selectbox(
        "Select which GPA to predict using SAT scores:",
        ("HSGPA", "GPA")
    )

    # Prepare Data
    X = data3[['SATV', 'SATM']]
    y = data3[gpa_option]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error (MSE): **{mse:.2f}**")
    st.write(f"R-squared (R²): **{r2:.2f}**")

    # Plot Predicted vs Actual GPA
    st.subheader("Predicted vs Actual GPA")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual GPA')
    ax.set_ylabel('Predicted GPA')
    ax.set_title(f'Actual vs Predicted {gpa_option}')
    st.pyplot(fig)

# Conclusion
st.header("Conclusion and Insights")
st.write("""
After exploring the datasets, we can draw several insights:
1. **Influence of Study Hours**: Study habits are crucial, but diminishing returns may occur at high study hours.
2. **Demographic Impact on GPA**: Factors like parental education, socioeconomic background, and gender may play a role in academic performance.
3. **Actionable Recommendations**: Insights like time management workshops, personalized tutoring, and early support for first-generation students could help improve GPA outcomes.

Future studies could incorporate larger datasets and track students over longer periods to confirm these findings.
""")
