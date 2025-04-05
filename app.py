import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
file_path = "student_data.csv"  # Update with correct file path
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoder = LabelEncoder()
df["Pass_Fail"] = label_encoder.fit_transform(df["Pass_Fail"])  # Pass = 1, Fail = 0
df["Gender"] = label_encoder.fit_transform(df["Gender"])  # Male = 1, Female = 0

# Define features and target variable
X = df.drop(columns=["Student_ID", "Final_Exam_Score", "Pass_Fail"])
y = df["Pass_Fail"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save trained model
joblib.dump(rf_model, "student_performance_model.pkl")

# Load the model
model = joblib.load("student_performance_model.pkl")

# Function to analyze weak subjects
def find_weak_subjects(student_scores):
    subjects = ["Physics", "Maths", "Biology", "Chemistry"]
    weak_subjects = [subject for subject in subjects if student_scores[subject] < 50]
    return weak_subjects

# Function to recommend YouTube videos based on weak subjects
def recommend_videos(weak_subjects):
    video_links = {
        "Physics": "https://www.youtube.com/results?search_query=physics+tutorial",
        "Maths": "https://www.youtube.com/results?search_query=maths+tutorial",
        "Biology": "https://www.youtube.com/results?search_query=biology+tutorial",
        "Chemistry": "https://www.youtube.com/results?search_query=chemistry+tutorial",
    }
    return {subject: video_links[subject] for subject in weak_subjects}

# Function to analyze study habits
def analyze_study_habits(social_media_hours, study_hours):
    if social_media_hours > 5:
        social_media_advice = "Reduce social media usage to improve concentration."
    else:
        social_media_advice = "Good balance of social media usage."

    if study_hours < 3:
        study_advice = "Increase study hours to at least 4-5 hours per day."
    else:
        study_advice = "Good study habits!"

    return social_media_advice, study_advice

# Function to provide well-being suggestions
def well_being_suggestions(screen_time, stress_level):
    suggestions = []
    if screen_time > 6:
        suggestions.append("Reduce screen time to avoid eye strain and improve focus.")
    if stress_level > 7:
        suggestions.append("Practice relaxation techniques like meditation or exercise.")
    
    return suggestions if suggestions else ["Good mental and physical health balance!"]

# Streamlit App Interface
st.title("🎓 Student Performance Prediction App")

st.sidebar.header("Enter Student Details")

# User Input Fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
attendance_rate = st.sidebar.slider("Attendance Rate (%)", 0, 100, 75)
social_media_hours = st.sidebar.slider("Daily Social Media Hours", 0, 10, 3)
study_hours = st.sidebar.slider("Daily Study Hours", 0, 10, 3)
physics = st.sidebar.slider("Physics Score", 0, 100, 50)
maths = st.sidebar.slider("Maths Score", 0, 100, 50)
biology = st.sidebar.slider("Biology Score", 0, 100, 50)
chemistry = st.sidebar.slider("Chemistry Score", 0, 100, 50)
past_exam_scores = st.sidebar.slider("Past Exam Scores", 0, 100, 60)
screen_time = st.sidebar.slider("Daily Screen Time (hours)", 0, 12, 5)
stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)

# Convert user input into model-compatible format
student_data = {
    "Gender": 1 if gender == "Male" else 0,
    "Attendance_Rate": attendance_rate,
    "Social_Media_Hours": social_media_hours,
    "Study_Hours_per_Day": study_hours,
    "Physics": physics,
    "Maths": maths,
    "Biology": biology,
    "Chemistry": chemistry,
    "Past_Exam_Scores": past_exam_scores
}

student_df = pd.DataFrame([student_data])

# Predict performance
if st.sidebar.button("Predict Performance"):
    predicted_result = model.predict(student_df)[0]
    result_text = "Pass ✅" if predicted_result == 1 else "Fail ❌"
    
    # Weak Subject Analysis
    weak_subjects = find_weak_subjects(student_data)
    video_recommendations = recommend_videos(weak_subjects)

    # Study habits analysis
    social_media_advice, study_advice = analyze_study_habits(social_media_hours, study_hours)

    # Well-being suggestions
    wellness_advice = well_being_suggestions(screen_time, stress_level)

    # Display Results
    st.subheader("🎯 Prediction Results")
    st.success(f"Performance Prediction: *{result_text}*")

    st.subheader("📌 Weak Subject Analysis")
    if weak_subjects:
        st.warning(f"Needs Improvement in: {', '.join(weak_subjects)}")
        st.subheader("📺 YouTube Video Recommendations")
        for subject, link in video_recommendations.items():
            st.markdown(f"- [{subject} Tutorial]({link})")
    else:
        st.success("Great job! No weak subjects detected.")

    st.subheader("📚 Study Habits Analysis")
    st.info(f"📵 {social_media_advice}")
    st.info(f"📖 {study_advice}")

    st.subheader("💆 Well-being Suggestions")
    for advice in wellness_advice:
        st.warning(advice)