import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load model and data
model = joblib.load("student_risk_model.pkl")  # Your trained RandomForestClassifier
data = pd.read_csv("student_dataset_with_ids.csv")
features = joblib.load("feature_names.pkl")    # Ensure feature list used during training is loaded

# Ensure StudentID is string
data['StudentID'] = data['StudentID'].astype(str)

# Sidebar Role Selection
st.sidebar.title("User Role")
role = st.sidebar.radio("Who are you?", ("Teacher", "Parent"))

# Sidebar Student Selection
students = data['StudentID'].unique()
selected_id = st.sidebar.selectbox("Select Student ID", students)

# Filter the data for selected student
student_df = data[data['StudentID'] == selected_id].copy()
X_input = student_df[features].values.reshape(1, -1)

# Prediction
risk_map = {0: "Low", 1: "Medium", 2: "High"}
predicted_risk = model.predict(X_input)[0]
predicted_label = risk_map[predicted_risk]

# Main Title
st.title("🎓 Student Performance Monitoring Dashboard")
st.subheader(f"🔍 Viewing data for Student ID: {selected_id}")

# Show Data Table
st.write("### 📋 Student Info")
st.dataframe(student_df.drop(columns=["StudentID"]))

# Prediction
st.success(f"✅ Predicted Risk Level: **{predicted_label}**")

# Parent View: Radar Chart for Subject Scores
if role == "Parent":
    st.write("### 📈 Subject-wise Radar Chart")
    subject_cols = ['HSC', 'SSC', 'English', 'Computer']

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=student_df[subject_cols].values.flatten(),
        theta=subject_cols,
        fill='toself',
        name='Scores'
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        title="Subject Performance Radar"
    )
    st.plotly_chart(radar_fig, use_container_width=True)

# Teacher View: Bar charts and distribution
if role == "Teacher":
    st.write("### 🧠 Academic Scores")
    st.bar_chart(student_df[['HSC', 'SSC', 'English', 'Computer']].T)

    st.write("### 📊 Attendance & Extra Curricular Activities")
    st.bar_chart(student_df[['Attendance', 'Extra']].T)

    st.write("### 📊 Overall GPA Distribution (All Students)")
    fig = px.histogram(data, x="Overall", nbins=10, title="Overall GPA Distribution")
    st.plotly_chart(fig, use_container_width=True)
