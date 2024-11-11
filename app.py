import streamlit as st
import sqlite3
import pandas as pd

# Connect to the SQLite database
def get_connection():
    conn = sqlite3.connect('movies.db')
    return conn

# Function to fetch data based on user input
def fetch_data(query):
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# Set up the Streamlit app
st.title("IMDB Movie Data Analysis")
st.write("This app analyzes key factors and relationships that may affect movie ratings.")

# Filter options
st.sidebar.header("Filter Options")
min_votes = st.sidebar.slider("Minimum Number of Votes", 0, 100000, 1000)
rating = st.sidebar.slider("Minimum Rating", 1.0, 10.0, 5.0)

# Query based on user selections
query = f'''
    SELECT * FROM movies
    WHERE numVotes >= {min_votes} AND averageRating >= {rating}
'''
data = fetch_data(query)

# Display the data
st.write("Filtered Data")
st.dataframe(data)

# Basic analysis and visualizations
st.write("Average Rating Distribution")
st.bar_chart(data['averageRating'].value_counts().sort_index())

# Additional analysis: e.g., rating distribution by genre
genre_count = data['genres'].value_counts().head(10)
st.write("Top 10 Genres by Rating Count")
st.bar_chart(genre_count)
