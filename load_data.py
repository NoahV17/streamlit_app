import sqlite3
import pandas as pd

# Load CSV data
data = pd.read_csv('data/movies_initial.csv')

# Connect to SQLite (or create a new database if it doesn’t exist)
conn = sqlite3.connect('movies.db')
cursor = conn.cursor()

# Create the table structure if it hasn’t been created
cursor.execute('''
    CREATE TABLE IF NOT EXISTS movies (
        id TEXT PRIMARY KEY,
        title TEXT,
        genres TEXT,
        averageRating REAL,
        numVotes INTEGER,
        releaseYear INTEGER
    );
''')

# Insert data in batches for efficiency
batch_size = 500
for i in range(0, len(data), batch_size):
    batch = data.iloc[i:i+batch_size]
    batch.to_sql('movies', conn, if_exists='append', index=False)

# Close the connection
conn.commit()
conn.close()
