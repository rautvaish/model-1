import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "student_data.csv"  # Update with correct path if needed
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

# Model evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("Classification Report:\n", classification_report(y_test, y_pred))

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

# Example Student Input (Fixed: Added Gender)
example_student = {
    "Gender": 1,  # Male = 1, Female = 0
    "Attendance_Rate": 75.0,
    "Social_Media_Hours": 6,
    "Study_Hours_per_Day": 2,
    "Physics": 45,
    "Maths": 70,
    "Biology": 40,
    "Chemistry": 50,
    "Past_Exam_Scores": 65
}

# Convert input to DataFrame (Ensure same column order as in training)
student_df = pd.DataFrame([example_student])

# Predict performance
predicted_result = rf_model.predict(student_df)[0]
result_text = "Pass" if predicted_result == 1 else "Fail"

# Find weak subjects
weak_subjects = find_weak_subjects(example_student)
video_recommendations = recommend_videos(weak_subjects)

# Study habits analysis
social_media_advice, study_advice = analyze_study_habits(example_student["Social_Media_Hours"], example_student["Study_Hours_per_Day"])

# Well-being suggestions
wellness_advice = well_being_suggestions(example_student["Social_Media_Hours"], 6)  # Assuming stress level 6

# Print Recommendations
print("\nStudent Performance Prediction:", result_text)
print("Weak Subjects:", weak_subjects)
print("YouTube Video Recommendations:", video_recommendations)
print("Study Habits Advice:", social_media_advice, "|", study_advice)
print("Well-being Suggestions:", wellness_advice)