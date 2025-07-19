import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = 'All_Streaming_Shows.csv'  # Update with your correct file path
shows_df = pd.read_csv(file_path)

# Handle infinite values by replacing them with NaN
shows_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Data Preprocessing
cleaned_df = shows_df.dropna(subset=['IMDB Rating', 'Genre', 'Streaming Platform', 'Content Rating']).copy()
cleaned_df['IMDB Rating'] = pd.to_numeric(cleaned_df['IMDB Rating'], errors='coerce')
cleaned_df.dropna(subset=['IMDB Rating'], inplace=True)

# Encoding and Scaling
le_genre = LabelEncoder()
le_platform = LabelEncoder()
cleaned_df['Genre Encoded'] = le_genre.fit_transform(cleaned_df['Genre'])
cleaned_df['Platform Encoded'] = le_platform.fit_transform(cleaned_df['Streaming Platform'])

scaler = StandardScaler()
cleaned_df['Scaled IMDb Rating'] = scaler.fit_transform(cleaned_df[['IMDB Rating']])

# Prepare features and target variable for Random Forest
cleaned_df['Target'] = cleaned_df['IMDB Rating'] >= 7.0
X = cleaned_df[['Genre Encoded', 'Platform Encoded', 'Scaled IMDb Rating']]
y = cleaned_df['Target'].astype(int)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building and training a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Recommendation function based on user input
def recommend_best_shows(age, genre, platform, min_imdb_rating=7.0, top_n=10):
    valid_ratings = get_valid_ratings(age)
   
    # Apply filtering based on age-restricted genres only for Romance
    filtered_df = cleaned_df[
        (cleaned_df['Content Rating'].isin(valid_ratings)) &
        (cleaned_df['Genre'].str.contains(genre, case=False, na=False)) &
        (cleaned_df['Streaming Platform'].str.contains(platform, case=False, na=False)) &
        (cleaned_df['IMDB Rating'] >= min_imdb_rating)
    ]
   
    # Check for age restrictions only for Romance
    if age < 18 and genre.lower() == 'romance':
        return "Sorry, the genre 'Romance' is restricted for your age group."
   
    top_shows = filtered_df.sort_values(by='IMDB Rating', ascending=False).head(top_n)
    return top_shows[['Series Title', 'Year Released', 'IMDB Rating', 'Genre', 'Streaming Platform']].reset_index(drop=True)

def get_valid_ratings(age):
    if age >= 18:
        return ['18+', '16+', '13+', '7+', 'All Ages']
    elif age >= 16:
        return ['16+', '13+', '7+', 'All Ages']
    elif age >= 13:
        return ['13+', '7+', 'All Ages']
    elif age >= 7:
        return ['7+', 'All Ages']
    else:
        return ['All Ages']

# Streamlit UI
st.title("🎬 Streaming Show Recommendations")
st.image("https://akm-img-a-in.tosshub.com/businesstoday/images/story/202306/ezgif-sixteen_nine_286.jpg?size=948:533", width=600)

st.markdown("""
    <style>
    .stTextInput {
        margin-bottom: 20px;
    }
    .stButton {
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

age = st.number_input("Enter your age:", min_value=0, max_value=100, value=18)

# Dropdown for genre selection
genre = st.selectbox("Select your preferred genre:", 
                     ["Action", "Romance", "Animation", "Thriller", "Horror", "Comedy", "Drama"])

platform = st.text_input("Enter your preferred streaming platform:")

tab_10, tab_20 = st.tabs(["Top 10 Shows", "Top 20 Shows"])

with tab_10:
    if st.button("Get Top 10 Shows"):
        if genre and platform:
            results = recommend_best_shows(age, genre, platform, top_n=10)
            if isinstance(results, str):
                st.write(results)
            elif results.empty:
                st.write("No shows found matching your criteria.")
            else:
                st.write(results)
        else:
            st.write("Please enter both genre and platform.")

with tab_20:
    if st.button("Get Top 20 Shows"):
        if genre and platform:
            results = recommend_best_shows(age, genre, platform, top_n=20)
            if isinstance(results, str):
                st.write(results)
            elif results.empty:
                st.write("No shows found matching your criteria.")
            else:
                st.write(results)
        else:
            st.write("Please enter both genre and platform.")
