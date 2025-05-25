import streamlit as st
import pandas as pd
import json
import os
import re
import tempfile
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import boto3
from rapidfuzz import process
from tqdm import tqdm

# Page configuration
st.set_page_config(
    page_title="Medical Entity Extraction & Adverse Event Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features_df' not in st.session_state:
    st.session_state.features_df = None
if 'adverse_events' not in st.session_state:
    st.session_state.adverse_events = set()

# Helper Classes and Functions
class TextCleaner:
    FILLER_WORDS = {
        "um", "uh", "like", "you know", "i mean", "so", "well", "hmm",
        "ah", "er", "eh", "okay", "right", "yeah", "huh", "basically"
    }
    
    @staticmethod
    def clean_text(text):
        text = text.lower()
        for word in TextCleaner.FILLER_WORDS:
            pattern = r'\b' + re.escape(word) + r'\b'
            text = re.sub(pattern, '', text)
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class FeatureExtractor:
    @staticmethod
    def extract_features(labeled_data):
        features = []
        for entry in labeled_data:
            if not isinstance(entry, dict):
                continue
            all_entities = entry.get("entities", [])
            features.append({
                "filename": entry.get("file", ""),
                "num_medications": sum(1 for e in all_entities if e.get("category") == "MEDICATION"),
                "num_symptoms": sum(1 for e in all_entities if e.get("category") == "SYMPTOM"),
                "num_procedures": sum(1 for e in all_entities if e.get("category") == "TEST_TREATMENT_PROCEDURE"),
                "num_adverse_events": sum(1 for e in all_entities if e.get("is_adverse_event") is True),
                "num_entities": len(all_entities),
                "has_adverse_event": any(e.get("is_adverse_event") is True for e in all_entities)
            })
        return features

def load_faers_adverse_events():
    """Simulate loading FAERS adverse events"""
    sample_adverse_events = [
        "nausea", "vomiting", "diarrhea", "headache", "dizziness", "fatigue",
        "rash", "allergic reaction", "chest pain", "shortness of breath",
        "abdominal pain", "constipation", "insomnia", "anxiety", "depression",
        "hypertension", "hypotension", "tachycardia", "bradycardia", "arrhythmia"
    ]
    return set(sample_adverse_events)

def label_entity(entity, adverse_event_set):
    """Label entity as adverse event or not"""
    entity_text_raw = entity.get("Text") or entity.get("text", "")
    entity_text = entity_text_raw.strip().lower()
    
    # Exact match
    if entity_text in adverse_event_set:
        return True
    
    # Fuzzy match
    if adverse_event_set:
        match, score, _ = process.extractOne(entity_text, adverse_event_set)
        return score >= 90
    return False

# Sidebar navigation
st.sidebar.markdown("# 🏥 Medical Analysis Pipeline")
page = st.sidebar.selectbox(
    "Select Analysis Step:",
    ["🏠 Home", "📄 Text Input & Preprocessing", "🧠 Entity Extraction", "🏷️ Entity Labeling", "📊 Feature Engineering", "🤖 Model Training", "📈 Analytics Dashboard"]
)

# Main content based on selected page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🏥 Medical Entity Extraction & Adverse Event Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Medical Text Analysis Pipeline
    
    This application provides a comprehensive solution for analyzing medical transcripts and detecting adverse events. 
    The pipeline consists of several integrated steps:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Pipeline Steps:
        1. **Text Input & Preprocessing** - Clean and prepare medical transcripts
        2. **Entity Extraction** - Extract medical entities using NLP
        3. **Entity Labeling** - Label entities as adverse events
        4. **Feature Engineering** - Create features for ML models
        5. **Model Training** - Train adverse event prediction models
        6. **Analytics Dashboard** - Visualize results and insights
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Key Features:
        - **Multi-format Support** - Process text files and manual input
        - **Medical NER** - Advanced medical entity recognition
        - **Adverse Event Detection** - FAERS-based adverse event matching
        - **Machine Learning** - Random Forest classification models
        - **Interactive Visualizations** - Comprehensive analytics dashboard
        - **Export Capabilities** - Download processed data and models
        """)
    
    st.markdown("""
    ### 🚀 Getting Started
    1. Navigate through the pipeline steps using the sidebar
    2. Start with **Text Input & Preprocessing** to upload or enter medical text
    3. Follow the sequential steps to complete the analysis
    4. View results in the **Analytics Dashboard**
    """)

elif page == "📄 Text Input & Preprocessing":
    st.markdown('<h2 class="section-header">📄 Text Input & Preprocessing</h2>', unsafe_allow_html=True)
    
    input_method = st.radio("Choose input method:", ["Manual Text Input", "File Upload"])
    
    if input_method == "Manual Text Input":
        st.subheader("Enter Medical Text")
        user_text = st.text_area(
            "Paste your medical transcript here:",
            height=200,
            placeholder="Enter medical text containing patient symptoms, medications, procedures, etc."
        )
        
        if st.button("Process Text") and user_text:
            with st.spinner("Processing text..."):
                cleaned_text = TextCleaner.clean_text(user_text)
                st.session_state.processed_data['raw_text'] = user_text
                st.session_state.processed_data['cleaned_text'] = cleaned_text
                
            st.markdown('<div class="success-box">✅ Text processed successfully!</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Text")
                st.text_area("", value=user_text, height=150, disabled=True)
            
            with col2:
                st.subheader("Cleaned Text")
                st.text_area("", value=cleaned_text, height=150, disabled=True)
    
    else:  # File Upload
        st.subheader("Upload Text Files")
        uploaded_files = st.file_uploader(
            "Choose text files",
            type=['txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Files"):
            processed_files = {}
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                content = str(file.read(), "utf-8")
                cleaned_content = TextCleaner.clean_text(content)
                processed_files[file.name] = {
                    'raw': content,
                    'cleaned': cleaned_content
                }
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.session_state.processed_data['files'] = processed_files
            st.markdown(f'<div class="success-box">✅ Processed {len(uploaded_files)} files successfully!</div>', unsafe_allow_html=True)
            
            # Display file processing summary
            st.subheader("Processing Summary")
            for filename, data in processed_files.items():
                with st.expander(f"📄 {filename}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text("Original Length:")
                        st.write(f"{len(data['raw'])} characters")
                    with col2:
                        st.text("Cleaned Length:")
                        st.write(f"{len(data['cleaned'])} characters")

elif page == "🧠 Entity Extraction":
    st.markdown('<h2 class="section-header">🧠 Medical Entity Extraction</h2>', unsafe_allow_html=True)
    
    if 'processed_data' not in st.session_state or not st.session_state.processed_data:
        st.markdown('<div class="warning-box">⚠️ Please process some text first in the Text Input & Preprocessing section.</div>', unsafe_allow_html=True)
    else:
        st.subheader("Extract Medical Entities")
        
        # Simulate AWS Comprehend Medical (since we can't use real AWS in this environment)
        def simulate_entity_extraction(text):
            """Simulate AWS Comprehend Medical entity extraction"""
            # Sample entities based on common medical terms
            entities = []
            
            # Medication patterns
            med_patterns = ['medication', 'drug', 'pill', 'tablet', 'prescription', 'aspirin', 'acetaminophen', 'ibuprofen']
            for pattern in med_patterns:
                if pattern.lower() in text.lower():
                    entities.append({
                        'Text': pattern,
                        'Category': 'MEDICATION',
                        'Type': 'GENERIC_NAME',
                        'Score': 0.95,
                        'BeginOffset': text.lower().find(pattern.lower()),
                        'EndOffset': text.lower().find(pattern.lower()) + len(pattern)
                    })
            
            # Symptom patterns
            symptom_patterns = ['pain', 'nausea', 'headache', 'fever', 'cough', 'fatigue', 'dizziness', 'rash']
            for pattern in symptom_patterns:
                if pattern.lower() in text.lower():
                    entities.append({
                        'Text': pattern,
                        'Category': 'SYMPTOM',
                        'Type': 'SIGN_OR_SYMPTOM',
                        'Score': 0.88,
                        'BeginOffset': text.lower().find(pattern.lower()),
                        'EndOffset': text.lower().find(pattern.lower()) + len(pattern)
                    })
            
            # Procedure patterns
            proc_patterns = ['surgery', 'procedure', 'test', 'examination', 'scan', 'x-ray', 'blood test']
            for pattern in proc_patterns:
                if pattern.lower() in text.lower():
                    entities.append({
                        'Text': pattern,
                        'Category': 'TEST_TREATMENT_PROCEDURE',
                        'Type': 'PROCEDURE_NAME',
                        'Score': 0.90,
                        'BeginOffset': text.lower().find(pattern.lower()),
                        'EndOffset': text.lower().find(pattern.lower()) + len(pattern)
                    })
            
            return entities
        
        if st.button("Extract Entities"):
            extracted_entities = {}
            
            # Process single text or multiple files
            if 'cleaned_text' in st.session_state.processed_data:
                with st.spinner("Extracting entities..."):
                    text = st.session_state.processed_data['cleaned_text']
                    entities = simulate_entity_extraction(text)
                    extracted_entities['single_text'] = entities
            
            elif 'files' in st.session_state.processed_data:
                progress_bar = st.progress(0)
                files = st.session_state.processed_data['files']
                
                for i, (filename, data) in enumerate(files.items()):
                    entities = simulate_entity_extraction(data['cleaned'])
                    extracted_entities[filename] = entities
                    progress_bar.progress((i + 1) / len(files))
            
            st.session_state.processed_data['entities'] = extracted_entities
            st.markdown('<div class="success-box">✅ Entity extraction completed!</div>', unsafe_allow_html=True)
            
            # Display extracted entities
            st.subheader("Extracted Entities Summary")
            
            total_entities = 0
            entity_categories = {}
            
            for source, entities in extracted_entities.items():
                total_entities += len(entities)
                for entity in entities:
                    category = entity.get('Category', 'Unknown')
                    entity_categories[category] = entity_categories.get(category, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entities", total_entities)
            with col2:
                st.metric("Medications", entity_categories.get('MEDICATION', 0))
            with col3:
                st.metric("Symptoms", entity_categories.get('SYMPTOM', 0))
            
            # Visualization
            if entity_categories:
                fig = px.pie(
                    values=list(entity_categories.values()),
                    names=list(entity_categories.keys()),
                    title="Entity Distribution by Category"
                )
                st.plotly_chart(fig)
            
            # Detailed view
            for source, entities in extracted_entities.items():
                with st.expander(f"📋 Entities from {source} ({len(entities)} entities)"):
                    if entities:
                        df = pd.DataFrame(entities)
                        st.dataframe(df[['Text', 'Category', 'Type', 'Score']])
                    else:
                        st.write("No entities found.")

elif page == "🏷️ Entity Labeling":
    st.markdown('<h2 class="section-header">🏷️ Entity Labeling for Adverse Events</h2>', unsafe_allow_html=True)
    
    if 'entities' not in st.session_state.processed_data:
        st.markdown('<div class="warning-box">⚠️ Please extract entities first in the Entity Extraction section.</div>', unsafe_allow_html=True)
    else:
        st.subheader("Load Adverse Events Database")
        
        # Load FAERS adverse events
        if st.button("Load FAERS Adverse Events") or st.session_state.adverse_events:
            if not st.session_state.adverse_events:
                st.session_state.adverse_events = load_faers_adverse_events()
            
            st.markdown(f'<div class="success-box">✅ Loaded {len(st.session_state.adverse_events)} adverse events from database</div>', unsafe_allow_html=True)
            
            # Display sample adverse events
            st.subheader("Sample Adverse Events")
            sample_events = list(st.session_state.adverse_events)[:20]
            st.write(", ".join(sample_events))
            
            if st.button("Label Entities"):
                labeled_entities = {}
                
                for source, entities in st.session_state.processed_data['entities'].items():
                    labeled_source_entities = []
                    for entity in entities:
                        entity_copy = entity.copy()
                        entity_copy['is_adverse_event'] = label_entity(entity, st.session_state.adverse_events)
                        labeled_source_entities.append(entity_copy)
                    labeled_entities[source] = labeled_source_entities
                
                st.session_state.processed_data['labeled_entities'] = labeled_entities
                st.markdown('<div class="success-box">✅ Entity labeling completed!</div>', unsafe_allow_html=True)
                
                # Display labeling results
                st.subheader("Labeling Results")
                
                total_adverse = 0
                total_entities = 0
                
                for source, entities in labeled_entities.items():
                    adverse_count = sum(1 for e in entities if e.get('is_adverse_event'))
                    total_adverse += adverse_count
                    total_entities += len(entities)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Entities", total_entities)
                with col2:
                    st.metric("Adverse Events", total_adverse)
                with col3:
                    st.metric("Adverse Event Rate", f"{(total_adverse/total_entities*100):.1f}%" if total_entities > 0 else "0%")
                
                # Detailed results
                for source, entities in labeled_entities.items():
                    adverse_entities = [e for e in entities if e.get('is_adverse_event')]
                    if adverse_entities:
                        with st.expander(f"🚨 Adverse Events in {source} ({len(adverse_entities)} found)"):
                            df = pd.DataFrame(adverse_entities)
                            st.dataframe(df[['Text', 'Category', 'Type', 'Score', 'is_adverse_event']])

elif page == "📊 Feature Engineering":
    st.markdown('<h2 class="section-header">📊 Feature Engineering</h2>', unsafe_allow_html=True)
    
    if 'labeled_entities' not in st.session_state.processed_data:
        st.markdown('<div class="warning-box">⚠️ Please complete entity labeling first.</div>', unsafe_allow_html=True)
    else:
        st.subheader("Extract Features for Machine Learning")
        
        if st.button("Extract Features"):
            # Convert labeled entities to the format expected by FeatureExtractor
            labeled_data = []
            for source, entities in st.session_state.processed_data['labeled_entities'].items():
                labeled_data.append({
                    'file': source,
                    'entities': entities
                })
            
            # Extract features
            features = FeatureExtractor.extract_features(labeled_data)
            features_df = pd.DataFrame(features)
            
            # Add derived features
            features_df["adverse_event_ratio"] = features_df["num_adverse_events"] / (features_df["num_entities"] + 1e-5)
            
            st.session_state.features_df = features_df
            st.markdown('<div class="success-box">✅ Feature extraction completed!</div>', unsafe_allow_html=True)
            
            # Display features
            st.subheader("Extracted Features")
            st.dataframe(features_df)
            
            # Feature statistics
            st.subheader("Feature Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numerical Summary:**")
                st.dataframe(features_df.describe())
            
            with col2:
                st.write("**Target Distribution:**")
                target_counts = features_df['has_adverse_event'].value_counts()
                fig = px.bar(
                    x=target_counts.index.map({True: "Has Adverse Event", False: "No Adverse Event"}),
                    y=target_counts.values,
                    title="Distribution of Adverse Event Cases"
                )
                st.plotly_chart(fig)
            
            # Feature correlations
            st.subheader("Feature Correlations")
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            corr_matrix = features_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig)
            
            # Download features
            csv = features_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Features CSV",
                data=csv,
                file_name="extracted_features.csv",
                mime="text/csv"
            )

elif page == "🤖 Model Training":
    st.markdown('<h2 class="section-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)
    
    if st.session_state.features_df is None:
        st.markdown('<div class="warning-box">⚠️ Please complete feature engineering first.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state.features_df
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Positive Cases", df['has_adverse_event'].sum())
        with col3:
            st.metric("Features", len(df.columns) - 2)  # Exclude filename and target
        
        # Model training options
        st.subheader("Training Configuration")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        
        if st.button("Train Model"):
            if len(df) < 4:
                st.error("❌ Not enough samples for training. Need at least 4 samples.")
            else:
                # Prepare data
                X = df.drop(columns=["filename", "has_adverse_event"])
                y = df["has_adverse_event"]
                
                # Check if we have both classes
                if len(y.unique()) < 2:
                    st.error("❌ Need samples from both classes (with and without adverse events) for training.")
                else:
                    try:
                        # Train/test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                        
                        # Train model
                        with st.spinner("Training model..."):
                            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                            model.fit(X_train, y_train)
                        
                        # Predictions
                        y_pred = model.predict(X_test)
                        
                        # Store model
                        st.session_state.model = model
                        
                        st.markdown('<div class="success-box">✅ Model training completed!</div>', unsafe_allow_html=True)
                        
                        # Model evaluation
                        st.subheader("Model Performance")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Accuracy Score:**")
                            accuracy = accuracy_score(y_test, y_pred)
                            st.metric("Accuracy", f"{accuracy:.3f}")
                            
                            st.write("**Classification Report:**")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                        
                        with col2:
                            st.write("**Confusion Matrix:**")
                            cm = confusion_matrix(y_test, y_pred)
                            fig = px.imshow(
                                cm,
                                title="Confusion Matrix",
                                labels=dict(x="Predicted", y="Actual"),
                                color_continuous_scale="Blues"
                            )
                            st.plotly_chart(fig)
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        fig = px.bar(
                            feature_importance,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Feature Importance"
                        )
                        st.plotly_chart(fig)
                        
                        # Rule-based baseline comparison
                        st.subheader("Baseline Comparison")
                        df["rule_based_prediction"] = df["adverse_event_ratio"] > 0.3
                        rule_accuracy = accuracy_score(df["has_adverse_event"], df["rule_based_prediction"])
                        
                        comparison_df = pd.DataFrame({
                            'Model': ['Rule-based Baseline', 'Random Forest'],
                            'Accuracy': [rule_accuracy, accuracy]
                        })
                        
                        fig = px.bar(
                            comparison_df,
                            x='Model',
                            y='Accuracy',
                            title="Model Comparison"
                        )
                        st.plotly_chart(fig)
                        
                    except ValueError as e:
                        st.error(f"❌ Error during training: {str(e)}")

elif page == "📈 Analytics Dashboard":
    st.markdown('<h2 class="section-header">📈 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.features_df is None:
        st.markdown('<div class="warning-box">⚠️ Please complete the pipeline steps to view analytics.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state.features_df
        
        # Key metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cases", len(df))
        with col2:
            st.metric("Adverse Events", df['has_adverse_event'].sum())
        with col3:
            avg_entities = df['num_entities'].mean()
            st.metric("Avg Entities/Case", f"{avg_entities:.1f}")
        with col4:
            avg_ratio = df['adverse_event_ratio'].mean()
            st.metric("Avg AE Ratio", f"{avg_ratio:.3f}")
        
        # Distribution plots
        st.subheader("📈 Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Entity counts distribution
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=('Medications', 'Symptoms', 'Procedures', 'Adverse Events'))
            
            fig.add_trace(go.Histogram(x=df['num_medications'], name='Medications'), row=1, col=1)
            fig.add_trace(go.Histogram(x=df['num_symptoms'], name='Symptoms'), row=1, col=2)
            fig.add_trace(go.Histogram(x=df['num_procedures'], name='Procedures'), row=2, col=1)
            fig.add_trace(go.Histogram(x=df['num_adverse_events'], name='Adverse Events'), row=2, col=2)
            
            fig.update_layout(title="Entity Count Distributions", showlegend=False)
            st.plotly_chart(fig)
        
        with col2:
            # Adverse event ratio distribution
            fig = px.histogram(
                df,
                x='adverse_event_ratio',
                title="Adverse Event Ratio Distribution",
                nbins=20
            )
            st.plotly_chart(fig)
            
            # Scatter plot: entities vs adverse events
            fig = px.scatter(
                df,
                x='num_entities',
                y='num_adverse_events',
                color='has_adverse_event',
                title="Total Entities vs Adverse Events",
                hover_data=['filename']
            )
            st.plotly_chart(fig)
        
        # Detailed case analysis
        st.subheader("🔍 Case Analysis")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_only_adverse = st.checkbox("Show only cases with adverse events")
        with col2:
            min_entities = st.slider("Minimum entities", 0, int(df['num_entities'].max()), 0)
        
        # Filter data
        filtered_df = df.copy()
        if show_only_adverse:
            filtered_df = filtered_df[filtered_df['has_adverse_event'] == True]
        filtered_df = filtered_df[filtered_df['num_entities'] >= min_entities]
        
        st.write(f"Showing {len(filtered_df)} cases")
        st.dataframe(filtered_df)
        
        # Export options
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Full Dataset"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="medical_analysis_results.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.session_state.model is not None:
                if st.button("💾 Download Trained Model"):
                    # Simulate model download (in real app, you'd use joblib.dump)
                    st.success("Model download initiated! (In real app, this would download the .pkl file)")
        
        # Summary insights
        st.subheader("💡 Key Insights")
        
        insights = []
        
        # Calculate insights
        adverse_rate = (df['has_adverse_event'].sum() / len(df)) * 100
        insights.append(f"**Adverse Event Rate**: {adverse_rate:.1f}% of cases contain adverse events")
        
        if df['has_adverse_event'].sum() > 0:
            avg_ae_per_case = df[df['has_adverse_event']]['num_adverse_events'].mean()
            insights.append(f"**Average AEs per positive case**: {avg_ae_per_case:.1f}")
        
        most_common_entity_type = df[['num_medications', 'num_symptoms', 'num_procedures']].sum().idxmax()
        entity_type_map = {
            'num_medications': 'Medications',
            'num_symptoms': 'Symptoms', 
            'num_procedures': 'Procedures'
        }
        insights.append(f"**Most common entity type**: {entity_type_map[most_common_entity_type]}")
        
        if len(df) > 1:
            correlation = df['num_entities'].corr(df['num_adverse_events'])
            insights.append(f"**Entity-AE correlation**: {correlation:.3f}")
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        # Model performance summary
        if st.session_state.model is not None:
            st.subheader("🎯 Model Performance Summary")
            st.markdown("""
            **Model Status**: ✅ Trained and ready for predictions  
            **Algorithm**: Random Forest Classifier  
            **Features Used**: Entity counts, adverse event ratios  
            **Use Case**: Predicting adverse events in medical transcripts
            """)
            
            # Prediction interface
            st.subheader("🔮 Make Predictions")
            st.markdown("Enter entity counts to predict adverse event probability:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                pred_medications = st.number_input("Medications", min_value=0, value=2)
            with col2:
                pred_symptoms = st.number_input("Symptoms", min_value=0, value=3)
            with col3:
                pred_procedures = st.number_input("Procedures", min_value=0, value=1)
            
            pred_adverse = st.number_input("Known Adverse Events", min_value=0, value=0)
            
            if st.button("🎯 Predict"):
                # Calculate derived features
                total_entities = pred_medications + pred_symptoms + pred_procedures + pred_adverse
                ae_ratio = pred_adverse / (total_entities + 1e-5)
                
                # Prepare input
                input_data = [[pred_medications, pred_symptoms, pred_procedures, pred_adverse, total_entities, ae_ratio]]
                
                # Make prediction
                prediction = st.session_state.model.predict(input_data)[0]
                probability = st.session_state.model.predict_proba(input_data)[0]
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", "Adverse Event" if prediction else "No Adverse Event")
                with col2:
                    st.metric("Confidence", f"{max(probability):.2%}")
                
                # Probability breakdown
                prob_df = pd.DataFrame({
                    'Outcome': ['No Adverse Event', 'Adverse Event'],
                    'Probability': probability
                })
                
                fig = px.bar(
                    prob_df,
                    x='Outcome',
                    y='Probability',
                    title="Prediction Probabilities",
                    color='Probability',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 Medical Entity Extraction & Adverse Event Detection Pipeline</p>
    <p>Built with Streamlit • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)