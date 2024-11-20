import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import matplotlib.pyplot as plt

# Title
st.title("Chronic Otitis Media Detection Mock")
st.write("Capture 3-5 images from the camera for analysis.")

# Session state for images
if "captured_images" not in st.session_state:
    st.session_state["captured_images"] = []

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None  # Initialize frame

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(self.frame, format="bgr24")

# WebRTC setup
webrtc_ctx = webrtc_streamer(
    key="camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Camera status
if webrtc_ctx and webrtc_ctx.state.playing:
    st.success("Camera is active. You can capture images.")
elif webrtc_ctx and not webrtc_ctx.state.playing:
    st.warning("Waiting for the camera to initialize...")

# Capture images
if webrtc_ctx and webrtc_ctx.video_processor:
    if st.button("Capture Image"):
        frame = webrtc_ctx.video_processor.frame
        if frame is not None:
            if len(st.session_state["captured_images"]) < 5:
                st.session_state["captured_images"].append(frame)
                st.success(f"Captured image {len(st.session_state['captured_images'])}")
            else:
                st.warning("You can only capture up to 5 images.")
        else:
            st.warning("No frame available. Try again.")

# Display captured images
if st.session_state["captured_images"]:
    st.image(
        st.session_state["captured_images"],
        caption=[f"Image {i+1}" for i in range(len(st.session_state["captured_images"]))],
        use_column_width=True,
    )

# Generate a mock report
if len(st.session_state["captured_images"]) >= 3:
    st.write("### Final Report (Mockup Data)")
    mock_confidences = [
        {"Ear Wax": 0.85, "Chronic Otitis Media": 0.60, "Acute Otitis Media": 0.30, "Healthy": 0.75},
        {"Ear Wax": 0.80, "Chronic Otitis Media": 0.65, "Acute Otitis Media": 0.40, "Healthy": 0.70},
        {"Ear Wax": 0.75, "Chronic Otitis Media": 0.55, "Acute Otitis Media": 0.35, "Healthy": 0.65},
    ]

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

    if st.button("Generate Report"):
        generate_report(mock_confidences)

# Clear captured images
if st.button("Clear Images"):
    st.session_state["captured_images"] = []
    st.success("Images cleared. Start over!")
