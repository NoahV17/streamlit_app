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
dataset_name = st.sidebar.selectbox("Select Dataset", ["Student Performance", "GPA and Study Hours", "First-Year GPA"])

# Define thesis and research questions
st.title("Exploring Factors Influencing Academic Success")
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
    1. How do students' math, reading, and writing scores correlate, and are there patterns based on demographics?
    2. Are there performance differences across demographic groups, such as gender or parental education level?
    """)

    # Data Visualization
    st.subheader("Correlation of Academic Scores")
    fig, ax = plt.subplots()
    sns.heatmap(data1[['math score', 'reading score', 'writing score']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Box Plot of Scores by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x=data1['gender'], y=data1['math score'], ax=ax)
    st.pyplot(fig)

elif dataset_name == "GPA and Study Hours":
    st.header("Dataset 2: GPA and Study Hours")
    st.write("Analyzing how study habits affect GPA.")

    # Basic Data Overview
    st.subheader("Data Preview")
    st.write(data2.head())

    # Questions to Answer
    st.subheader("Research Questions")
    st.write("""
    1. What is the correlation between study hours and GPA, and is there a threshold where study hours significantly boost GPA?
    2. Are there diminishing returns for study hours in terms of GPA?
    """)

    # Data Visualization
    st.subheader("Study Hours vs. GPA")
    fig, ax = plt.subplots()
    sns.scatterplot(x=data2['study_hours'], y=data2['GPA'], ax=ax)
    st.pyplot(fig)

    # Model Training and Prediction
    st.subheader("Modeling GPA Based on Study Hours")
    X = data2[['study_hours']]
    y = data2['GPA']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    st.write("Mean Squared Error (Linear Regression):", mean_squared_error(y_test, y_pred_lr))
    st.write("R2 Score (Linear Regression):", r2_score(y_test, y_pred_lr))

elif dataset_name == "First-Year GPA":
    st.header("Dataset 3: First-Year GPA")
    st.write("Examining how demographic factors influence first-year college GPA.")

    # Basic Data Overview
    st.subheader("Data Preview")
    st.write(data3.head())

    # Questions to Answer
    st.subheader("Research Questions")
    st.write("""
    1. How do demographic factors like age, gender, and background influence first-year GPA?
    2. Is there a significant difference in GPA among students from different socioeconomic backgrounds?
    """)

    # Data Visualization
    st.subheader("Distribution of First-Year GPA by Demographic")
    fig, ax = plt.subplots()
    sns.histplot(data3['first_year_gpa'], kde=True, ax=ax)
    st.pyplot(fig)

    # Modeling GPA based on demographics (Example with selected features)
    st.subheader("Predicting First-Year GPA Based on Demographics")
    selected_features = ['age', 'gender', 'high_school_gpa']  # Replace with relevant columns
    X = data3[selected_features]
    y = data3['first_year_gpa']
    X = pd.get_dummies(X, drop_first=True)  # Encoding categorical variables if necessary
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    st.write("Mean Squared Error (Random Forest):", mean_squared_error(y_test, y_pred_rf))
    st.write("R2 Score (Random Forest):", r2_score(y_test, y_pred_rf))

# Conclusion
st.header("Conclusion and Insights")
st.write("""
After exploring the datasets, we can draw several insights:
1. **Influence of Study Hours**: Study habits are crucial, but diminishing returns may occur at high study hours.
2. **Demographic Impact on GPA**: Factors like parental education, socioeconomic background, and gender may play a role in academic performance.
3. **Actionable Recommendations**: Insights like time management workshops, personalized tutoring, and early support for first-generation students could help improve GPA outcomes.

Future studies could incorporate larger datasets and track students over longer periods to confirm these findings.
""")
