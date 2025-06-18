import streamlit as st
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import time
import os
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Initialize session state for user data and real-time monitoring
if 'user_data' not in st.session_state:
    st.session_state.user_data = []
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'monitoring_data' not in st.session_state:
    st.session_state.monitoring_data = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# Load model, scaler, and feature names with error handling
try:
    model = joblib.load('pcos_model.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('feature_names.json', 'r') as f:
        FEATURE_NAMES = json.load(f)
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# Function to load dataset
def load_dataset():
    try:
        # Try to load the dataset from the same directory as the model
        dataset_path = 'PCOS_data.csv'
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load dataset
if st.session_state.dataset is None:
    st.session_state.dataset = load_dataset()

# Page config
st.set_page_config(
    page_title="PCOS/PCOD Detection System",
    page_icon="🏥",
    layout="wide"
)

# Main content
st.title("PCOS/PCOD Detection System")
st.write("""
This app predicts the risk of Polycystic Ovary Syndrome (PCOS) or Polycystic Ovarian Disease (PCOD) based on your input.
**Note:** This is for educational/screening purposes only. Always consult a healthcare professional for diagnosis.
""")

# Function to generate medical suggestions
def generate_medical_suggestions(user_data):
    suggestions = {
        'lifestyle': [],
        'diet': [],
        'exercise': [],
        'medical': [],
        'monitoring': []
    }
    
    # BMI-based suggestions
    if user_data['bmi'] > 25:
        suggestions['lifestyle'].append("Focus on weight management through a balanced diet and regular exercise")
        suggestions['diet'].append("Consider consulting a nutritionist for a personalized diet plan")
        suggestions['exercise'].append("Aim for 150 minutes of moderate exercise per week")
    elif user_data['bmi'] < 18.5:
        suggestions['lifestyle'].append("Focus on healthy weight gain through proper nutrition")
        suggestions['diet'].append("Increase caloric intake with nutrient-dense foods")
    
    # Cycle regularity suggestions
    if user_data['cycle_regularity'] <= 2:
        suggestions['medical'].append("Consider tracking your menstrual cycle using a calendar or app")
        suggestions['monitoring'].append("Monitor and record any changes in cycle length or symptoms")
    
    # Symptom-based suggestions
    if user_data['hair_growth'] >= 3:
        suggestions['medical'].append("Consult a dermatologist for hirsutism management")
    if user_data['skin_darkening'] >= 3:
        suggestions['medical'].append("Consider consulting a dermatologist for acanthosis nigricans")
    if user_data['hair_loss'] >= 3:
        suggestions['medical'].append("Consult a dermatologist for hair loss treatment")
    if user_data['pimples'] >= 3:
        suggestions['medical'].append("Consider consulting a dermatologist for acne management")
    
    # Lifestyle-based suggestions
    if user_data['fast_food'] >= 3:
        suggestions['diet'].append("Reduce fast food consumption and focus on home-cooked meals")
    if user_data['exercise'] <= 2:
        suggestions['exercise'].append("Start with light exercises like walking or yoga")
    
    # General suggestions
    suggestions['lifestyle'].append("Maintain a regular sleep schedule")
    suggestions['diet'].append("Stay hydrated and limit caffeine intake")
    suggestions['exercise'].append("Include both cardio and strength training in your routine")
    suggestions['medical'].append("Schedule regular check-ups with your healthcare provider")
    suggestions['monitoring'].append("Keep a symptom diary to track changes")
    
    return suggestions

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Prediction", "Real-time Monitoring", "History", "Dataset Analysis", "Medical Suggestions", "Resources"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            age = st.number_input("Age", min_value=12, max_value=60, value=25)
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=22.0, step=0.1)
            cycle_regularity = st.selectbox("Cycle Regularity", [1,2,3,4,5], 
                format_func=lambda x: ["Very Irregular","Irregular","Moderate","Regular","Very Regular"][x-1])
            weight_gain = st.selectbox("Weight Gain", [1,2,3,4,5], 
                format_func=lambda x: ["None","Slight","Moderate","Significant","Severe"][x-1])
        
        with col2:
            st.subheader("Symptoms")
            hair_growth = st.selectbox("Hair Growth", [1,2,3,4,5], 
                format_func=lambda x: ["None","Slight","Moderate","Significant","Severe"][x-1])
            skin_darkening = st.selectbox("Skin Darkening", [1,2,3,4,5], 
                format_func=lambda x: ["None","Slight","Moderate","Significant","Severe"][x-1])
            hair_loss = st.selectbox("Hair Loss", [1,2,3,4,5], 
                format_func=lambda x: ["None","Slight","Moderate","Significant","Severe"][x-1])
            pimples = st.selectbox("Pimples", [1,2,3,4,5], 
                format_func=lambda x: ["None","Slight","Moderate","Significant","Severe"][x-1])
        
        st.subheader("Lifestyle")
        col3, col4 = st.columns(2)
        with col3:
            fast_food = st.selectbox("Fast Food Consumption", [1,2,3,4,5], 
                format_func=lambda x: ["Never","Rarely","Sometimes","Often","Very Often"][x-1])
        with col4:
            exercise = st.selectbox("Exercise Frequency", [1,2,3,4,5], 
                format_func=lambda x: ["Never","Rarely","Sometimes","Often","Very Often"][x-1])
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create DataFrame with proper feature names
        features = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'cycle_regularity': [cycle_regularity],
            'weight_gain': [weight_gain],
            'hair_growth': [hair_growth],
            'skin_darkening': [skin_darkening],
            'hair_loss': [hair_loss],
            'pimples': [pimples],
            'fast_food': [fast_food],
            'exercise': [exercise]
        }, columns=FEATURE_NAMES)
        
        # Scale features
        features_scaled = scaler.transform(features)
        features_scaled_df = pd.DataFrame(features_scaled, columns=FEATURE_NAMES)
        
        # Make prediction with error handling
        try:
            prediction = model.predict_proba(features_scaled_df)[0]
            probability = prediction[1] * 100
            risk = "High Risk" if probability > 50 else "Low Risk"
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")
            st.stop()

        # Save prediction to history
        prediction_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'age': age,
            'bmi': bmi,
            'cycle_regularity': cycle_regularity,
            'weight_gain': weight_gain,
            'hair_growth': hair_growth,
            'skin_darkening': skin_darkening,
            'hair_loss': hair_loss,
            'pimples': pimples,
            'fast_food': fast_food,
            'exercise': exercise,
            'risk': risk,
            'probability': probability
        }
        st.session_state.user_data.append(prediction_data)
        st.session_state.last_prediction = prediction_data

        # Display results
        st.markdown("### Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### Risk Level: {risk}")
            st.markdown(f"**Probability:** {probability:.2f}%")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Feature importance visualization using model's feature_importances_
            importance_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title='Feature Importance in Prediction')
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Real-time Monitoring")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Monitoring"):
            st.session_state.monitoring_active = True
            if not st.session_state.monitoring_data and st.session_state.last_prediction:
                st.session_state.monitoring_data = [st.session_state.last_prediction.copy()]
    with col2:
        if st.button("Pause Monitoring"):
            st.session_state.monitoring_active = False
    with col3:
        if st.button("Resume Monitoring"):
            if st.session_state.monitoring_data:
                st.session_state.monitoring_active = True

    if st.session_state.monitoring_active:
        st.success("Monitoring active - collecting data...")
        chart_placeholder = st.empty()
        alert_placeholder = st.empty()
        table_placeholder = st.empty()
        control_placeholder = st.empty()

        # Always use the last value in monitoring_data as the base
        if st.session_state.monitoring_data:
            last_data = st.session_state.monitoring_data[-1].copy()

            # --- Interactive Controls ---
            with control_placeholder.container():
                st.markdown("#### Adjust Features in Real Time")
                new_bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=float(last_data['bmi']), step=0.1)
                new_exercise = st.slider("Exercise Frequency", min_value=1, max_value=5, value=int(last_data['exercise']), step=1)
                # Update last_data with user input
                last_data['bmi'] = new_bmi
                last_data['exercise'] = new_exercise

            # Simulate new data point
            last_data['bmi'] += np.random.normal(0, 0.1)
            last_data['probability'] += np.random.normal(0, 1)
            last_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.monitoring_data.append(last_data)

            if len(st.session_state.monitoring_data) > 1:
                monitoring_df = pd.DataFrame(st.session_state.monitoring_data)
                fig1 = px.line(monitoring_df, x='timestamp', y='probability', title='Real-time Risk Probability')
                chart_placeholder.plotly_chart(fig1, use_container_width=True)

                # Alerts and metrics as before
                if len(monitoring_df) > 2:
                    last_prob = monitoring_df['probability'].iloc[-1]
                    prev_prob = monitoring_df['probability'].iloc[-2]
                    change = last_prob - prev_prob
                    if abs(change) > 5:
                        alert_placeholder.warning(f"Significant change detected: {change:.1f}%")
                    elif change > 0:
                        alert_placeholder.info(f"Risk increasing: +{change:.1f}%")
                    else:
                        alert_placeholder.success(f"Risk decreasing: {change:.1f}%")

                st.subheader("Current Metrics")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Current Risk", f"{last_data['probability']:.1f}%")
                with metrics_col2:
                    st.metric("BMI", f"{last_data['bmi']:.1f}")
                with metrics_col3:
                    st.metric("Risk Level", last_data['risk'])

                # --- Live Feature Table ---
                feature_table = pd.DataFrame([last_data])[[
                    'age', 'bmi', 'cycle_regularity', 'weight_gain', 'hair_growth',
                    'skin_darkening', 'hair_loss', 'pimples', 'fast_food', 'exercise',
                    'probability', 'risk', 'timestamp']]
                table_placeholder.dataframe(feature_table.T, use_container_width=True, height=400)

            time.sleep(5)
            st.experimental_rerun()
        else:
            st.info("No prediction data to monitor. Please make a prediction first.")
    else:
        st.info("Monitoring is inactive. Click 'Start Monitoring' to begin real-time analysis.")

with tab3:
    st.header("Prediction History")
    if st.session_state.user_data:
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.user_data)
        
        # Display history table
        st.dataframe(history_df)
        
        # Plot trend
        if len(history_df) > 1:
            fig = px.line(history_df, x='timestamp', y='probability',
                         title='Risk Probability Trend Over Time')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add correlation analysis
            st.subheader("Correlation Analysis")
            correlation_df = history_df[['bmi', 'cycle_regularity', 'weight_gain', 
                                      'hair_growth', 'skin_darkening', 'hair_loss', 
                                      'pimples', 'fast_food', 'exercise', 'probability']]
            corr_matrix = correlation_df.corr()
            fig = px.imshow(corr_matrix,
                          title='Feature Correlations with Risk Probability',
                          color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction history available yet.")

with tab4:
    st.header("Dataset Analysis")
    
    if st.session_state.dataset is not None:
        # Dataset Overview
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(st.session_state.dataset))
        with col2:
            st.metric("PCOS Cases", len(st.session_state.dataset[st.session_state.dataset['PCOS'] == 1]))
        with col3:
            st.metric("Non-PCOS Cases", len(st.session_state.dataset[st.session_state.dataset['PCOS'] == 0]))
        
        # Age Distribution
        st.subheader("Age Distribution")
        fig_age = px.histogram(st.session_state.dataset, x='Age', color='PCOS',
                             title='Age Distribution by PCOS Status',
                             labels={'Age': 'Age (years)', 'count': 'Number of Cases'},
                             color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        st.plotly_chart(fig_age, use_container_width=True)
        
        # BMI Analysis
        st.subheader("BMI Analysis")
        fig_bmi = px.box(st.session_state.dataset, x='PCOS', y='BMI',
                        title='BMI Distribution by PCOS Status',
                        labels={'BMI': 'Body Mass Index', 'PCOS': 'PCOS Status'})
        st.plotly_chart(fig_bmi, use_container_width=True)
        
        # Symptom Analysis
        st.subheader("Symptom Analysis")
        symptoms = ['Hair_growth', 'Skin_darkening', 'Hair_loss', 'Pimples']
        symptom_data = []
        
        for symptom in symptoms:
            pcos_cases = st.session_state.dataset[st.session_state.dataset['PCOS'] == 1][symptom].mean()
            non_pcos_cases = st.session_state.dataset[st.session_state.dataset['PCOS'] == 0][symptom].mean()
            symptom_data.append({
                'Symptom': symptom.replace('_', ' '),
                'PCOS Cases': pcos_cases,
                'Non-PCOS Cases': non_pcos_cases
            })
        
        symptom_df = pd.DataFrame(symptom_data)
        fig_symptoms = px.bar(symptom_df, x='Symptom', y=['PCOS Cases', 'Non-PCOS Cases'],
                            title='Average Symptom Severity by PCOS Status',
                            barmode='group')
        st.plotly_chart(fig_symptoms, use_container_width=True)
        
        # Lifestyle Factors
        st.subheader("Lifestyle Factors")
        lifestyle = ['Fast_food', 'Exercise']
        lifestyle_data = []
        
        for factor in lifestyle:
            pcos_cases = st.session_state.dataset[st.session_state.dataset['PCOS'] == 1][factor].mean()
            non_pcos_cases = st.session_state.dataset[st.session_state.dataset['PCOS'] == 0][factor].mean()
            lifestyle_data.append({
                'Factor': factor.replace('_', ' '),
                'PCOS Cases': pcos_cases,
                'Non-PCOS Cases': non_pcos_cases
            })
        
        lifestyle_df = pd.DataFrame(lifestyle_data)
        fig_lifestyle = px.bar(lifestyle_df, x='Factor', y=['PCOS Cases', 'Non-PCOS Cases'],
                             title='Lifestyle Factors by PCOS Status',
                             barmode='group')
        st.plotly_chart(fig_lifestyle, use_container_width=True)
        
        # Correlation Analysis
        st.subheader("Feature Correlations")
        numeric_cols = st.session_state.dataset.select_dtypes(include=[np.number]).columns
        corr_matrix = st.session_state.dataset[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix,
                           title='Feature Correlations',
                           color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Compare with User Data
        if st.session_state.user_data:
            st.subheader("Compare with Your Data")
            user_df = pd.DataFrame(st.session_state.user_data)
            
            # Age Comparison
            fig_age_comp = go.Figure()
            fig_age_comp.add_trace(go.Box(y=st.session_state.dataset['Age'], name='Population'))
            fig_age_comp.add_trace(go.Box(y=user_df['age'], name='Your Data'))
            fig_age_comp.update_layout(title='Age Distribution Comparison')
            st.plotly_chart(fig_age_comp, use_container_width=True)
            
            # BMI Comparison
            fig_bmi_comp = go.Figure()
            fig_bmi_comp.add_trace(go.Box(y=st.session_state.dataset['BMI'], name='Population'))
            fig_bmi_comp.add_trace(go.Box(y=user_df['bmi'], name='Your Data'))
            fig_bmi_comp.update_layout(title='BMI Distribution Comparison')
            st.plotly_chart(fig_bmi_comp, use_container_width=True)
    else:
        st.warning("Dataset not found. Please ensure 'PCOS_data.csv' is in the same directory as the application.")

with tab5:
    st.header("Medical Suggestions")
    
    if st.session_state.last_prediction:
        suggestions = generate_medical_suggestions(st.session_state.last_prediction)
        
        # Display suggestions in expandable sections
        with st.expander("Lifestyle Recommendations", expanded=True):
            for suggestion in suggestions['lifestyle']:
                st.write(f"• {suggestion}")
        
        with st.expander("Dietary Recommendations", expanded=True):
            for suggestion in suggestions['diet']:
                st.write(f"• {suggestion}")
        
        with st.expander("Exercise Recommendations", expanded=True):
            for suggestion in suggestions['exercise']:
                st.write(f"• {suggestion}")
        
        with st.expander("Medical Recommendations", expanded=True):
            for suggestion in suggestions['medical']:
                st.write(f"• {suggestion}")
        
        with st.expander("Monitoring Recommendations", expanded=True):
            for suggestion in suggestions['monitoring']:
                st.write(f"• {suggestion}")
        
        # Additional Resources
        st.subheader("Additional Resources")
        st.markdown("""
        ### Recommended Healthcare Providers
        - Endocrinologist: For hormonal management
        - Gynecologist: For reproductive health
        - Dermatologist: For skin and hair concerns
        - Nutritionist: For dietary guidance
        
        ### Important Notes
        - These suggestions are based on your input data and general PCOS management guidelines
        - Always consult with healthcare professionals before making significant lifestyle changes
        - Keep track of your symptoms and report any changes to your healthcare provider
        - Regular check-ups are essential for monitoring your condition
        """)
        
        # Emergency Information
        st.warning("""
        ### Emergency Warning Signs
        If you experience any of the following, seek immediate medical attention:
        - Severe abdominal pain
        - Heavy bleeding
        - Difficulty breathing
        - Chest pain
        - Severe headaches
        """)
    else:
        st.info("Please make a prediction first to receive personalized medical suggestions.")

with tab6:
    st.header("Resources and Information")
    st.markdown("""
    ### Understanding PCOS/PCOD
    
    **What is PCOS/PCOD?**
    Polycystic Ovary Syndrome (PCOS) or Polycystic Ovarian Disease (PCOD) is a hormonal disorder common among women of reproductive age.
    
    **Common Symptoms:**
    - Irregular menstrual cycles
    - Excess hair growth
    - Acne
    - Weight gain
    - Hair loss
    
    **Management Tips:**
    1. Maintain a healthy diet
    2. Regular exercise
    3. Stress management
    4. Regular medical check-ups
    
    **Important Note:**
    This tool is for educational purposes only. Always consult with healthcare professionals for proper diagnosis and treatment.
    """)
    
    # Add external resources
    st.markdown("### External Resources")
    st.markdown("""
    - [PCOS Awareness Association](https://www.pcosaa.org/)
    - [National Institute of Health - PCOS](https://www.nichd.nih.gov/health/topics/pcos)
    - [PCOS Challenge](https://pcoschallenge.org/)
    """) 