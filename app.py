import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from ultralytics import YOLO
import os
import torch
import time

# Set page config
st.set_page_config(
    page_title="Fire & Smoke Detector",
    page_icon="🔥",
    layout="wide"
)

# App title and description
st.title("🔥 Fire and Smoke Detection System")
st.markdown("Upload an image/video or use your webcam to detect fire and smoke")

# Load the model
@st.cache_resource
def load_model():
    # Check multiple possible locations for the model
    model_paths = [
        "models/fire_smoke_detector/weights/best.pt",
        "runs/detect/train/weights/best.pt",
        "yolo11n.pt"  # Fallback to base model
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            st.sidebar.success(f"Using model: {path}")
            return YOLO(path)
    
    st.error("Model file not found.")
    st.stop()

model = load_model()

# Sidebar configuration
st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.1)

# Function to process images
def process_image(image, conf):
    results = model(image, conf=conf)
    return results[0].plot(), results[0].boxes

# Function to process videos - Fixed to handle file permissions properly
def process_video(video_file, conf):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile_name = tfile.name
        # Write video data to the temporary file
        tfile.write(video_file.read())
    
    # Open the video file with OpenCV
    vid_cap = cv2.VideoCapture(tfile_name)
    stframe = st.empty()
    
    try:
        while vid_cap.isOpened():
            success, frame = vid_cap.read()
            if not success:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = model(frame_rgb, conf=conf)
            annotated_frame = results[0].plot()
            
            # Display the frame
            stframe.image(annotated_frame, caption='Detected Fire/Smoke', use_container_width=True)
    except Exception as e:
        st.error(f"Error processing video: {e}")
    finally:
        # Always release the video capture object
        vid_cap.release()
        
        # Try to delete the file, with a retry mechanism for Windows
        try:
            os.unlink(tfile_name)
        except PermissionError:
            # On Windows, sometimes we need to wait before the file can be deleted
            time.sleep(1)
            try:
                os.unlink(tfile_name)
            except:
                st.warning("Could not delete temporary file. It will be removed later.")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Webcam Detection"])

# Tab 1: Image Detection
with tab1:
    st.header("Image Detection")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if image_file is not None:
        img = Image.open(image_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(img, use_container_width=True)
            
        with col2:
            st.subheader("Detection Result")
            with st.spinner("Processing image..."):
                annotated_img, boxes = process_image(img, confidence)
                st.image(annotated_img, use_container_width=True)
            
            # Display detection details
            if len(boxes) > 0:
                st.subheader("Detection Details")
                for box in boxes:
                    cls_id = int(box.cls.item())
                    cls_name = model.names[cls_id]
                    conf = box.conf.item()
                    st.write(f"- {cls_name}: {conf:.2f} confidence")
            else:
                st.write("No fire or smoke detected in this image.")
    else:
        # Show sample prediction if available
        if os.path.exists("models/sample_predictions.png"):
            st.image("models/sample_predictions.png", caption="Sample predictions", use_container_width=True)

# Tab 2: Video Detection
with tab2:
    st.header("Video Detection")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")
    
    if video_file is not None:
        st.video(video_file)
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                # Create a copy of the video file to avoid issues
                video_bytes = video_file.getvalue()
                video_file.seek(0)  # Reset file pointer after reading
                
                # Create a new temporary file-like object
                import io
                video_stream = io.BytesIO(video_bytes)
                process_video(video_stream, confidence)
            st.success("Video processing completed!")

# Tab 3: Webcam Detection
with tab3:
    st.header("Webcam Detection")
    st.write("Note: Webcam detection works only when running the app locally.")
    
    col1, col2 = st.columns(2)
    run_webcam = col1.button("Start Webcam")
    stop_webcam = col2.button("Stop Webcam")
    stframe = st.empty()
    
    if run_webcam and not stop_webcam:
        try:
            cap = cv2.VideoCapture(0)
            
            while cap.isOpened() and not stop_webcam:
                success, frame = cap.read()
                if not success:
                    st.error("Failed to read from webcam")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference
                results = model(frame_rgb, conf=confidence)
                annotated_frame = results[0].plot()
                
                # Display the frame
                stframe.image(annotated_frame, channels="RGB", use_container_width=True)
                
                # Check if stop button has been pressed
                if st.session_state.get('stop_pressed', False):
                    break
                    
            cap.release()
        except Exception as e:
            st.error(f"Error accessing webcam: {e}")

# Add a callback for stop button to set session state
if stop_webcam:
    st.session_state['stop_pressed'] = True

# Display device information in the sidebar
st.sidebar.markdown("---")
device = "GPU" if torch.cuda.is_available() else "CPU"
st.sidebar.write(f"Running on: **{device}**")
if torch.cuda.is_available():
    st.sidebar.write(f"GPU: **{torch.cuda.get_device_name(0)}**")

# Footer
st.markdown("---")
st.caption("Fire and Smoke Detection System using YOLOv11")