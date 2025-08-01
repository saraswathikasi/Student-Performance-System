import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load model and data
model = joblib.load("model.joblib")  # Trained RandomForestClassifier
data = pd.read_csv("data.csv")

# Ensure StudentID is string
data['StudentID'] = data['StudentID'].astype(str)

# Sidebar Role Selection
st.sidebar.title("User Role")
role = st.sidebar.radio("Who are you?", ("Teacher", "Parent"))

# Sidebar Student Selection
students = data['StudentID'].unique()
selected_id = st.sidebar.selectbox("Select Student ID", students)

# Filter the data for selected student
student_df = data[data['StudentID'] == selected_id]

st.title("🎓 Student Performance Monitoring Dashboard")
st.subheader(f"🔍 Viewing data for Student ID: {selected_id}")

# Display student data
st.write("### Student Info")
st.dataframe(student_df.drop("StudentID", axis=1))

# Prediction (Assumes 'Overall' is the target)
features = student_df.drop(columns=["StudentID", "Overall"])
prediction = model.predict(features)[0]

st.success(f"✅ Predicted Overall Performance: **{prediction}**")

# Radar chart (for Parent)
if role == "Parent":
    st.write("### 📈 Subject-wise Radar Chart")
    subject_cols = ['HSC', 'SSC', 'English', 'Computer']  # Add or modify as needed

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=student_df[subject_cols].values.flatten(),
        theta=subject_cols,
        fill='toself',
        name='Subject Scores'
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )
    st.plotly_chart(radar_fig)

# Bar charts (for Teacher)
if role == "Teacher":
    st.write("### 📊 Attendance & Extra Activities")
    st.bar_chart(student_df[['Attendance', 'Extra']].T)

    st.write("### 🧠 Academic Subjects Bar Chart")
    st.bar_chart(student_df[['HSC', 'SSC', 'English', 'Computer']].T)

    st.write("### 📊 Overall Distribution Across Students")
    fig = px.histogram(data, x="Overall", nbins=10, title="Overall Performance Distribution")
    st.plotly_chart(fig)
