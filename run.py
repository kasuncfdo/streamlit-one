import streamlit as st
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Title and instructions
st.title("Chronic Otitis Media Detection Mock")
st.write("Capture 3-5 images from the camera for analysis.")

# Initialize session state to store images
if 'captured_images' not in st.session_state:
    st.session_state['captured_images'] = []

# Video processor class to capture frames from the camera
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None
    
    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(self.frame, format="bgr24")

# Start the camera
webrtc_ctx = webrtc_streamer(
    key="camera", 
    video_processor_factory=VideoProcessor, 
    media_stream_constraints={"video": True, "audio": False}
)

# Displaying video feed status
if webrtc_ctx.video_processor is None:
    st.warning("Waiting for the camera to start...")
else:
    st.success("Camera is active. You can capture images.")

# Capture button to save frames
if webrtc_ctx.video_processor:
    if st.button("Capture Image"):
        if len(st.session_state['captured_images']) < 5:
            frame = webrtc_ctx.video_processor.frame
            if frame is not None:
                st.session_state['captured_images'].append(frame)
                st.success(f"Captured image {len(st.session_state['captured_images'])}")
            else:
                st.warning("No frame available to capture. Ensure the camera is active.")
        else:
            st.warning("You can only capture up to 5 images.")

# Display captured images
if st.session_state['captured_images']:
    st.image(
        st.session_state['captured_images'], 
        caption=[f"Image {i+1}" for i in range(len(st.session_state['captured_images']))], 
        use_column_width=True
    )

# Ensure the user has captured 3-5 images before proceeding to the mockup report
if len(st.session_state['captured_images']) >= 3:

    # Mockup data for confidence levels (skip API for now)
    mock_confidences = [
        {"Ear Wax": 0.85, "Chronic Otitis Media": 0.60, "Acute Otitis Media": 0.30, "Healthy": 0.75},
        {"Ear Wax": 0.80, "Chronic Otitis Media": 0.65, "Acute Otitis Media": 0.40, "Healthy": 0.70},
        {"Ear Wax": 0.75, "Chronic Otitis Media": 0.55, "Acute Otitis Media": 0.35, "Healthy": 0.65},
    ]

    # Generate mockup report with 4-axis chart
    def generate_report(confidence_data):
        fig, ax = plt.subplots()

        states = ["Ear Wax", "Chronic Otitis Media", "Acute Otitis Media", "Healthy"]
        x_labels = [f"Image {i+1}" for i in range(len(confidence_data))]

        for state in states:
            y_values = [confidence_data[i][state] for i in range(len(confidence_data))]
            ax.plot(x_labels, y_values, label=state)

        ax.set_xlabel("Images")
        ax.set_ylabel("Confidence Levels")
        ax.set_title("Confidence Levels by Condition")
        ax.legend()

        st.pyplot(fig)

    # Button to show the mockup report
    if st.button("Generate Report"):
        st.write("### Final Report (Mockup Data)")
        generate_report(mock_confidences)

else:
    st.warning("Please capture 3 to 5 images to proceed.")

# Allow clearing of captured images for a new submission
if st.button("Clear Images"):
    st.session_state['captured_images'] = []
    st.success("Images cleared. You can start capturing new ones.")
