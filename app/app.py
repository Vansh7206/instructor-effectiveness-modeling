import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Page Configuration
st.set_page_config(page_title="Instructor Effectiveness Predictor",page_icon="🎓",layout="centered")

#Loading saved Models
model = joblib.load("app/random_forest_model.pkl")
scaler = joblib.load("app/scaler.pkl")

#Page Title and Markdown
st.title("🎓 Instructor Effectiveness Predictor")
st.markdown("Enter batch-level metrics below to predict whether an instructor is **Low**, **Medium**, or **High** performer.")
st.divider()

#User Input area
st.subheader("📋 Enter Batch Metrics")

col1, col2 = st.columns(2)

with col1:
    completion_rate = st.slider("Completion Rate", 0.0, 1.0, 0.6, 0.01)
    dropout_rate = st.slider("Dropout Rate", 0.0, 1.0, 0.3, 0.01)
    avg_score_improvement = st.slider("Avg Score Improvement", 0.0, 50.0, 20.0, 0.5)
    avg_quiz_score = st.slider("Avg Quiz Score", 40.0, 100.0, 75.0, 0.5)
    avg_watch_time = st.slider("Avg Watch Time", 0.0, 1.0, 0.7, 0.01)
    assignment_submission_rate = st.slider("Assignment Submission Rate", 0.0, 1.0, 0.75, 0.01)

with col2:
    forum_activity_rate = st.slider("Forum Activity Rate", 0.0, 1.0, 0.25, 0.01)
    avg_feedback_score = st.slider("Avg Feedback Score", 1.0, 5.0, 4.0, 0.1)
    feedback_response_rate = st.slider("Feedback Response Rate", 0.0, 1.0, 0.6, 0.01)

st.divider()

#Button
if st.button("🔍 Predict Instructor Tier", use_container_width=True):

    #Dataframe
    raw_input = pd.DataFrame([[
        completion_rate, avg_score_improvement, avg_quiz_score,
        avg_watch_time, assignment_submission_rate, forum_activity_rate,
        avg_feedback_score, feedback_response_rate, dropout_rate
    ]], columns=[
        'completion_rate', 'avg_score_improvement', 'avg_quiz_score',
        'avg_watch_time', 'assignment_submission_rate', 'forum_activity_rate',
        'avg_feedback_score', 'feedback_response_rate', 'dropout_rate'
    ])

    scaled = scaler.transform(raw_input)
    scaled_df = pd.DataFrame(scaled, columns=raw_input.columns)

    #Dropout Special Case
    scaled_df['dropout_rate_inv'] = 1 - scaled_df['dropout_rate']
    scaled_df['dropout_rate_inv'] = scaled_df['dropout_rate_inv'].clip(lower=0)

    #Applying Rules
    scaled_df['pillar_learning'] = (
        scaled_df['completion_rate']       * 0.30 +
        scaled_df['dropout_rate_inv']      * 0.25 +
        scaled_df['avg_score_improvement'] * 0.25 +
        scaled_df['avg_quiz_score']        * 0.20
    )
    scaled_df['pillar_engagement'] = (
        scaled_df['avg_watch_time']             * 0.40 +
        scaled_df['assignment_submission_rate'] * 0.40 +
        scaled_df['forum_activity_rate']        * 0.20
    )
    scaled_df['pillar_quality'] = (
        scaled_df['avg_feedback_score']     * 0.60 +
        scaled_df['feedback_response_rate'] * 0.40
    )

    #Special Features
    feature_cols = [
        'pillar_learning', 'pillar_engagement', 'pillar_quality',
        'completion_rate', 'avg_quiz_score', 'avg_score_improvement',
        'dropout_rate_inv', 'avg_watch_time', 'assignment_submission_rate',
        'forum_activity_rate', 'avg_feedback_score', 'feedback_response_rate'
    ]

    #Default
    scaled_df['num_batches'] = 1

    feature_cols_with_batches = feature_cols + ['num_batches']
    X_input = scaled_df[feature_cols_with_batches]

    #Prediction
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    classes = model.classes_

    #Displaying results
    st.subheader("📊 Prediction Result")

    if prediction == "High":
        st.success(f"✅ Predicted Tier: **{prediction}**")
    elif prediction == "Medium":
        st.warning(f"⚠️ Predicted Tier: **{prediction}**")
    else:
        st.error(f"❌ Predicted Tier: **{prediction}**")

    #Confidence scores
    st.markdown("**Confidence Scores:**")
    prob_df = pd.DataFrame({
        'Tier': classes,
        'Confidence': [f"{p*100:.1f}%" for p in probabilities]
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    #Pillar Score 
    st.divider()
    st.subheader("🔍 Pillar Score Breakdown")

    col1, col2, col3 = st.columns(3)
    col1.metric("Learning Outcome", f"{scaled_df['pillar_learning'].values[0]:.3f}")
    col2.metric("Engagement", f"{scaled_df['pillar_engagement'].values[0]:.3f}")
    col3.metric("Instructor Quality", f"{scaled_df['pillar_quality'].values[0]:.3f}")

#Footer
st.divider()
st.caption("Built by Vansh Chandan | Instructor Effectiveness Modeling — EdTech")