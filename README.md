#  PulsePredict: Adverse Medical Event Prediction from Doctor-Patient Calls

##  Problem Statement

Adverse medical events often arise from miscommunication or overlooked information during patient-physician interactions. Identifying these events proactively could significantly improve patient safety and healthcare outcomes. However, there is currently **no scalable, automated method** to analyze spoken medical conversations and flag potential adverse outcomes.

**Deployed Link** - [PulsePredict App](https://pulsepredict01.streamlit.app/)

##  Objective

Develop an **end-to-end system** that:

- Transcribes doctor-patient audio conversations
- Extracts key medical entities
- Predicts the likelihood of an adverse medical event based on the conversation



##  Proposed Solution

PulsePredict follows a multi-stage processing pipeline:

1. **Audio Transcription**  
   Transcribe audio calls using OpenAI’s [`Whisper`](https://github.com/openai/whisper) model to convert speech into text.

2. **Medical Entity Extraction**  
   Use **AWS Comprehend Medical** to extract relevant medical entities (symptoms, medications, diagnoses, etc.) from the transcriptions.

3. **Labeling for Adverse Events**  
   Utilize a curated **FAERS (FDA Adverse Event Reporting System)** dataset to identify and label potential adverse medical events based on extracted entities.

4. **Feature Engineering**  
   Engineer structured features from the medical concepts to be used for model training.

5. **Adverse Event Prediction**  
   Train a **machine learning model** on these features to predict the probability of an adverse medical event occurring as a result of the doctor-patient interaction.



##  Workflow

The following diagram illustrates the complete pipeline for predicting adverse medical events from audio-based medical conversations:

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1vCYRadDdLqB6XdcZAeLKoc00I2lCE94I" alt="Adverse Event Detection Workflow" width="300"/>
</p>



##  Pipeline Scripts

A breakdown of the core scripts used in this project:

- **`predict_from_audio.py`**  
  Orchestrates the complete pipeline: from audio input to final prediction using the trained model.

- **`train_model.py`**  
  Trains the machine learning model using features extracted from labeled medical conversations.

- **`evaluate_model.py`**  
  Evaluates the trained model and benchmarks its performance against a rule-based baseline system.

- **`utils/`**  
  A directory containing helper functions for:
  - Loading FAERS adverse event data
  - Labeling entities with known adverse events
  - Parsing and cleaning entities from AWS Comprehend Medical

## How It Works

1. Transcribe medical calls using Whisper
2. Extract medical entities using AWS Comprehend Medical
3. Label entities based on FAERS adverse events database
4. Engineer features from labeled entities
5. Train a classifier to predict if an adverse event occurred
6. Run end-to-end predictions on new audio
 

## Demo Video

**Video Link** - [PulsePredict Video](https://drive.google.com/file/d/1bAoyQM2SPmOdzFpTUTmJmiLh9zBTr88n/view?usp=drive_link)

## Tech Stack


|  **Layer**            |  **Technology**               |
|------------------------|----------------------------------|
|  Transcription        | OpenAI Whisper                   |
|  NLP                 | AWS Comprehend Medical            |
|  ML Model            | Scikit-learn (Random Forest)      |
|  Backend             | Python                            |
|  Front End            | Streamlit                        |
|  Deployment           |                                  |


##  Testing

The project includes two types of testing:

- **UI Automation Testing**
  - Tests built with Playwright and Pytest
  - Validates UI elements: titles, input fields, and buttons
  - Runs in a headless browser (Chromium)
  - Ensures UI consistency and responsiveness

- **Manual Testing**
  - Covers backend pipeline from audio input to prediction
  - Tests transcription, entity extraction, and adverse event detection
  - Includes edge case handling (e.g. missing/corrupted files)
  - Helps verify functional correctness of each module

 **Test Documents**
- [UI Testing Documentation](./ui_tests/UI_Testing_Documentation.pdf)
- [Manual Testing Documentation](./manual_tests/Manual_Testing_Documentation.pdf)


## Challenges Faced & Solutions

### 1. Audio Transcription
- **Challenge**: Batch processing of .mp3 files with Whisper.
- **Issues**: Manual transcription, CPU performance warnings.
- **Solution**: Developed `batch_transcribe.py` and `predict_from_audio.py` for automated transcription.

### 2. Entity Extraction
- **Challenge**: Structured medical data extraction via AWS Comprehend Medical.
- **Issues**: Missing AWS credentials.
- **Solution**: Configured AWS CLI and used `batch_entity_extraction.py`.

### 3. Data Preprocessing
- **Challenge**: Noisy transcripts reduced NLP performance.
- **Solution**: Built `batch_preprocess_transcripts.py` to clean transcripts using a filler word list.

### 4. Adverse Event Labeling
- **Challenge**: Matching entities with FAERS data.
- **Issues**: Complex CSV format, exact matching.
- **Solution**: Cleaned FAERS data and used partial/lowercase matching in `label_entities.py`.

### 5. Feature Engineering
- **Challenge**: Poor model performance due to weak features.
- **Solution**: Added meaningful features (e.g., adverse_event_ratio) and rebuilt `feature_engineering.py`.

### 6. Model Training & Evaluation
- **Challenge**: Model overfitting and poor generalization.
- **Solution**: Balanced dataset with false samples, evaluated with both model and rule-based methods.

### 7. Model Bias Fix
- **Challenge**: Dataset bias towards positive samples.
- **Solution**: Added negative samples and improved feature diversity for better model performance.

### 8. Streamlit Deployment
- **Challenge**: Interface bugs and missing dependencies.
- **Solution**: Installed necessary libraries and finalized `medical_streamlit_app_updated.py`.

### 9. GitHub Cleanup
- **Challenge**: Uploaded unnecessary files, missing `.gitignore`.
- **Solution**: Added `.gitignore`, removed unused scripts, and updated project documentation.




## Screenshots

![Image5](https://github.com/user-attachments/assets/fca2aa89-6288-4858-a0e5-8a93f1c3ebbb)

![Image6](https://github.com/user-attachments/assets/b47b72c3-1ab7-4bf7-8da0-144840daf3d2)

![Image7](https://github.com/user-attachments/assets/f53ca72a-2179-433f-b83c-f921a8245ee6)

![Image8](https://github.com/user-attachments/assets/ada29165-ceb2-4574-9966-07cefcacd08f)

![Image9](https://github.com/user-attachments/assets/24c6f39d-6487-4dcb-b89c-8bb35a441707)

![Image10](https://github.com/user-attachments/assets/12b0c68c-c9a8-4279-b3bd-8d81a720ddba)

![Image11](https://github.com/user-attachments/assets/ac67cfb8-b0c8-4c28-ba7a-159ec6bd4567)

![Image12](https://github.com/user-attachments/assets/e37e1d1b-82c5-4073-af05-6d47077e4ca9)

![Image13](https://github.com/user-attachments/assets/0a332d33-5cba-4c55-b170-8d575f2bbd31)

![Image14](https://github.com/user-attachments/assets/dbd4b2c0-52e9-4af4-92c8-6119011dabf3)

![Image15](https://github.com/user-attachments/assets/1330ed26-8690-41d4-8552-8a274ee44dbf)

![Image16](https://github.com/user-attachments/assets/99cbddd7-3639-4c82-b91e-aa099b7e20da)

![Image17](https://github.com/user-attachments/assets/b032988b-cff4-4090-9e1b-1472bf208b09)

## Future Improvements
- Incorporate time-aware features such as event sequences and timestamps.  
- Use larger Whisper models to improve transcription accuracy.  
- Fine-tune domain-specific NLP models like BioBERT for better entity extraction.  
- Expand the FAERS dataset to cover more entity types and adverse events.















