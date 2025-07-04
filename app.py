import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import os
from playsound import playsound
import threading
import time
from PIL import Image

# --- Configuration Constants ---
MODEL_PATH = 'gender_detection_model.h5'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
ALERT_SOUND_PATH = 'alert.mp3'

# --- Global/Cached Resources ---
@st.cache_resource
def load_gender_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        st.warning("Please ensure 'gender_detection_model.h5' is in the same directory.")
        return None

@st.cache_resource
def load_face_cascade(path):
    try:
        face_cascade = cv2.CascadeClassifier(path)
        if face_cascade.empty():
            raise IOError(f"Could not load Haar Cascade classifier from {path}")
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face cascade from {path}: {e}")
        return None

def preprocess_face_for_model(face_roi, target_height, target_width):
    face_resized = cv2.resize(face_roi, (target_width, target_height))
    face_normalized = face_resized / 255.0
    face_preprocessed = np.expand_dims(face_normalized, axis=0)
    return face_preprocessed

def play_alert_sound():
    if not os.path.exists(ALERT_SOUND_PATH):
        st.warning(f"Alert sound file not found at {ALERT_SOUND_PATH}")
        return
    try:
        threading.Thread(target=playsound, args=(ALERT_SOUND_PATH,), daemon=True).start()
    except Exception as e:
        st.error(f"Failed to play alert sound: {e}")

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Women Safety Analytics System",
    page_icon="👩‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Header ---
st.markdown("""
<div style="background-color:#6a0dad;padding:10px;border-radius:10px">
<h1 style="color:white;text-align:center;">WOMEN SAFETY ANALYTICS SYSTEM</h1>
<p style="color:white;text-align:center;">Real-time gender detection for enhanced safety</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/female-profile.png", width=80)
    st.title("Navigation")
    
    page_options = {
        "👁️ Real-time Detection": "webcam_detection",
        "📚 Safety Resources": "safety_instructions",
        "ℹ️ About the Project": "about_project",
        "👥 Team Information": "team_details"
    }
    
    selected_page = st.radio(
        "Select Page",
        list(page_options.keys()),
        index=0
    )
    
    st.session_state.current_page = page_options[selected_page]
    
    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("🆘 Emergency Contacts"):
        st.session_state.show_emergency = True
    
    if st.session_state.get('show_emergency', False):
        with st.expander("Emergency Contacts", expanded=True):
            st.markdown("""
            - **Police**: 100
            - **Women's Helpline**: 1091
            - **Ambulance**: 108
            - **Domestic Abuse**: 181
            """)
            st.markdown("*Save these numbers in your phone*")

# --- Load Models ---
gender_model = load_gender_model(MODEL_PATH)
face_cascade = load_face_cascade(FACE_CASCADE_PATH)

if gender_model is None or face_cascade is None:
    st.error("Critical components failed to load. Please check the error messages above.")
    st.stop()

# Gender labels
class_labels = {0: 'Male', 1: 'Female'}

# --- Page Content ---
main_container = st.container()

if st.session_state.current_page == "webcam_detection":
    with main_container:
        st.markdown("## 🎥 Real-time Gender Detection")
        st.markdown("""
        This system analyzes webcam feed in real-time to detect gender composition and identify potential safety risks.
        When a lone female is detected with multiple males, the system triggers an alert.
        """)
        
        # Stats columns
        col1, col2, col3 = st.columns(3)
        with col1:
            faces_metric = st.metric("Faces Detected", "0", help="Total faces currently detected")
        with col2:
            female_metric = st.metric("Female Count", "0", help="Number of females detected")
        with col3:
            male_metric = st.metric("Male Count", "0", help="Number of males detected")
        
        # Webcam control
        col_start, col_stop, col_settings = st.columns([1, 1, 2])
        with col_start:
            start_webcam = st.button("▶️ Start Detection", key="start_btn", type="primary")
        with col_stop:
            stop_webcam = st.button("⏹️ Stop Detection", key="stop_btn")
        
        # Settings expander
        with col_settings:
            with st.expander("⚙️ Detection Settings"):
                detection_confidence = st.slider("Confidence Threshold", 0.7, 1.0, 0.8, 0.01)
                alert_scenario = st.selectbox(
                    "Alert Scenario",
                    ["1 Female + 2+ Males", "1 Female + 3+ Males"],
                    index=0
                )
        
        # Webcam feed placeholder
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        # Initialize session state for webcam
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        
        # Handle webcam control
        if start_webcam:
            st.session_state.webcam_active = True
        
        if stop_webcam:
            st.session_state.webcam_active = False
            frame_placeholder.empty()
            alert_placeholder.empty()
        
        # Webcam processing
        if st.session_state.webcam_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access webcam. Please check permissions.")
                st.session_state.webcam_active = False
            else:
                with st.spinner("Processing webcam feed..."):
                    while st.session_state.webcam_active and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame")
                            break
                        
                        display_frame = frame.copy()
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray_frame, 
                            scaleFactor=1.1, 
                            minNeighbors=5, 
                            minSize=(60, 60))
                        
                        female_count = 0
                        male_count = 0
                        
                        if len(faces) == 0:
                            cv2.putText(display_frame, "No faces detected", (50, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            for (x, y, w, h) in faces:
                                # Draw rectangle
                                color = (255, 105, 180)  # Pink
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                                
                                # Extract and process face
                                face_roi = frame[y:y+h, x:x+w]
                                if face_roi.size == 0:
                                    continue
                                
                                preprocessed_face = preprocess_face_for_model(face_roi, IMAGE_HEIGHT, IMAGE_WIDTH)
                                prediction_prob = gender_model.predict(preprocessed_face, verbose=0)[0][0]
                                predicted_class_idx = int(round(prediction_prob))
                                predicted_gender = class_labels.get(predicted_class_idx, "Unknown")
                                
                                # Update counts
                                if predicted_gender == 'Female':
                                    female_count += 1
                                    gender_color = (255, 0, 255)  # Purple
                                else:
                                    male_count += 1
                                    gender_color = (0, 255, 255)  # Yellow
                                
                                # Display gender and confidence
                                confidence = prediction_prob if predicted_class_idx == 1 else (1 - prediction_prob)
                                if confidence >= detection_confidence:
                                    text = f"{predicted_gender} ({confidence:.0%})"
                                    cv2.putText(display_frame, text, (x, y-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, gender_color, 2)
                        
                        # Check alert condition
                        alert_condition = (female_count == 1 and male_count >= 2) if alert_scenario == "1 Female + 2+ Males" else (female_count == 1 and male_count >= 3)
                        
                        if alert_condition:
                            alert_placeholder.error("""
                            ⚠️ **Potential Safety Risk Detected!**  
                            Lone female detected with multiple males.  
                            Please ensure safety measures are in place.
                            """)
                            play_alert_sound()
                            # Add red border to frame
                            display_frame = cv2.copyMakeBorder(
                                display_frame, 10, 10, 10, 10, 
                                cv2.BORDER_CONSTANT, value=[0, 0, 255]
                            )
                        else:
                            alert_placeholder.empty()
                        
                        # Update metrics
                        col1.metric("Faces Detected", len(faces))
                        col2.metric("Female Count", female_count)
                        col3.metric("Male Count", male_count)
                        
                        # Display frame
                        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
                        
                        # Small delay
                        time.sleep(0.05)
                    
                    cap.release()
                    st.session_state.webcam_active = False

elif st.session_state.current_page == "team_details":
    with main_container:
        st.markdown("## 👥 Our Team")
        
        # Team members in columns
        cols = st.columns(4)
        team_members = [
            {"name": "Polineni Keerthi Sri", "id": "99220040965", "role": "Developer"},
            {"name": "Shaik Gali Shahi", "id": "99220040993", "role": "Developer"},
            {"name": "Shaik Rafi", "id": "99220040996", "role": "Developer"},
            {"name": "Siddartha Rasani", "id": "99220040180", "role": "Developer"}
        ]
        
        for i, member in enumerate(team_members):
            with cols[i]:
                st.markdown(f"""
                <div style='background-color:#f0f2f6;border-radius:10px;padding:20px;text-align:center;box-shadow:0 4px 8px 0 rgba(0,0,0,0.1)'>
                    <img src='https://img.icons8.com/fluency/96/user-male-circle.png' width='60'><br>
                    <strong>{member['name']}</strong><br>
                    <small>ID: {member['id']}</small><br>
                    <small>Role: {member['role']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Advisor section
        st.markdown("---")
        st.markdown("### 🎓 Project Advisor")
        st.markdown("""
        <div style='background-color:#f8f9fa;border-radius:10px;padding:20px;box-shadow:0 4px 8px 0 rgba(0,0,0,0.1)'>
            <div style='display:flex;align-items:center;'>
                <img src='https://img.icons8.com/color/96/teacher.png' width='80'>
                <div style='margin-left:20px;'>
                    <strong>Dr. MURUGESWARI R</strong><br>
                    <em>Project Guide</em><br>
                    Department of Computer Science<br>
                    [University/College Name]
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.current_page == "safety_instructions":
    with main_container:
        st.markdown("## 📚 Women's Safety Resources")
        
        tab1, tab2, tab3 = st.tabs(["Safety Tips", "Self-Defense", "Emergency Planning"])
        
        with tab1:
            st.markdown("""
            ### 🛡️ Personal Safety Guidelines
            
            **General Awareness:**
            - Be aware of your surroundings at all times
            - Avoid walking alone at night in poorly lit areas
            - Trust your instincts - if something feels wrong, act on it
            
            **Technology Safety:**
            - Share your live location with trusted contacts when traveling
            - Use safety apps with emergency alert features
            - Keep your phone charged and accessible
            
            **Public Transportation:**
            - Sit near the driver or in well-populated areas
            - Avoid isolated bus stops or train stations
            - Have your ride arranged before leaving a location
            """)
        
        with tab2:
            st.markdown("""
            ### 🥋 Basic Self-Defense Techniques
            
            **1. Verbal Assertiveness:**
            - Use a strong, loud voice to set boundaries
            - Practice phrases like "Back off!" or "I don't know you!"
            
            **2. Physical Techniques:**
            - Target vulnerable areas: eyes, nose, throat, groin, knees
            - Use your elbows and knees - they're your strongest natural weapons
            - Practice breaking common grips and holds
            
            **3. Everyday Objects as Weapons:**
            - Keys between fingers for striking
            - Pepper spray (where legal)
            - Umbrella or water bottle can be used for defense
            
            *Consider taking a professional self-defense class for proper training*
            """)
        
        with tab3:
            st.markdown("""
            ### 🚨 Emergency Preparedness
            
            **Emergency Contacts:**
            - Save local emergency numbers in your phone
            - Program quick-dial shortcuts for emergency contacts
            
            **Safety Network:**
            - Establish a check-in system with friends/family
            - Create code words to discreetly signal distress
            
            **Escape Planning:**
            - Identify safe places along your regular routes
            - Know multiple exits from buildings you frequent
            - Keep your car keys handy (can be used as alarm trigger)
            
            **Digital Safety:**
            - Enable emergency SOS features on your smartphone
            - Share your location with trusted contacts when in unfamiliar areas
            """)

elif st.session_state.current_page == "about_project":
    with main_container:
        st.markdown("## ℹ️ About This Project")
        
        st.markdown("""
        ### 🌟 Project Overview
        
        The **Women Safety Analytics System** is an innovative application of computer vision and machine learning 
        technologies to enhance personal safety. By analyzing real-time video feeds, the system detects gender 
        composition in groups and identifies potentially risky scenarios based on configurable parameters.
        """)
        
        with st.expander("🔧 Technical Details", expanded=True):
            st.markdown("""
            **System Architecture:**
            - **Frontend**: Streamlit web application
            - **Computer Vision**: OpenCV for face detection
            - **Machine Learning**: Custom CNN model for gender classification
            - **Alert System**: Visual and auditory notifications
            
            **Key Features:**
            - Real-time video processing with low latency
            - Configurable detection thresholds
            - Responsive alert system
            - Detailed analytics dashboard
            """)
        
        with st.expander("📊 Performance Metrics", expanded=False):
            st.markdown("""
            - **Accuracy**: 92% on validation set
            - **Processing Speed**: 15-20 FPS on standard hardware
            - **Detection Range**: 1-5 meters from camera
            """)
        
        with st.expander("🚀 Future Enhancements", expanded=False):
            st.markdown("""
            Planned improvements include:
            - Integration with IoT safety devices
            - Mobile application version
            - Cloud-based alert notification system
            - Enhanced model with age detection
            """)
        
        st.markdown("""
        ### 📜 Project Documentation
        
        [Download Technical Specification](#) | [View Source Code](#) | [API Documentation](#)
        """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 10px; color: #666;">
    <p>© 2023 Women Safety Analytics System | For demonstration purposes only</p>
    <p>This project was developed as part of academic research</p>
</div>
""", unsafe_allow_html=True)