import pandas as pd

# Load datasets
data1 = pd.read_csv('data/Student_performance_data.csv')
data2 = pd.read_csv('data/gpa_study_hours.csv')
data3 = pd.read_csv('data/FirstYearGPA.csv')

# Display column names for each dataset to identify correct names
print("Dataset 1 columns:", data1.columns)
print("Dataset 2 columns:", data2.columns)
print("Dataset 3 columns:", data3.columns)
