import os
import time
import logging
import tempfile
import cv2 as cv
from PIL import Image
import streamlit as st
from ultralytics import YOLO

MODEL_DIR = './runs/detect/train/weights/best.pt'

logging.basicConfig(
    filename="./logs/log.log",
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)


def main():
    # Load a model
    global model
    model = YOLO(MODEL_DIR)

    st.sidebar.header("**Animal Classes**")

    class_names = ['Buffalo', 'Elephant', 'Rhino', 'Zebra', "Cheetah", "Fox", "Jaguar", "Tiger", "Lion", "Panda"]

    for animal in class_names:
        st.sidebar.markdown(f"- *{animal.capitalize()}*")

    st.title("Real-time Animal Species Detection")
    st.write("The aim of this project is to develop an efficient computer vision model capable of real-time wildlife detection.")

    # Load image or video
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4'])

    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            inference_images(uploaded_file)

        if uploaded_file.type.startswith('video'):
            inference_video(uploaded_file)


def inference_images(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    # Predict the image
    try:
        predict = model.predict(image)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        logging.error(f"Prediction error: {e}")
        return

    # Plot boxes
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    # Open the image
    st.image(plotted, caption="Detected Image", width=600)
    logging.info("Detected Image")


def inference_video(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv.VideoCapture(temp_file.name)
    fps = cap.get(cv.CAP_PROP_FPS)  # Get the frames per second of the video
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    current_frame = 0  # Track the current frame
    playing = False  # Start with pause

    if not cap.isOpened():
        st.error("Error opening video file.")

    # Create placeholders for play, pause, forward, backward buttons, and video display
    play_button = st.button("Play")
    pause_button = st.button("Pause")
    forward_button = st.button("Forward 5 seconds")
    backward_button = st.button("Backward 5 seconds")
    frame_placeholder = st.empty()
    
    # To stop the video
    stop_button = st.button("Stop")

    while cap.isOpened():
        if stop_button:
            os.unlink(temp_file.name)
            break

        # Check if play is pressed
        if play_button:
            playing = True

        # Check if pause is pressed
        if pause_button:
            playing = False

        # Check if forward button is pressed
        if forward_button:
            current_frame += int(fps * 5)  # Move forward by 5 seconds
            current_frame = min(current_frame, frame_count - 1)  # Limit to max frame count

        # Check if backward button is pressed
        if backward_button:
            current_frame -= int(fps * 5)  # Move backward by 5 seconds
            current_frame = max(current_frame, 0)  # Limit to minimum frame index

        # If playing, process video frames
        if playing:
            cap.set(cv.CAP_PROP_POS_FRAMES, current_frame)  # Set to current frame
            ret, frame = cap.read()

            if not ret:
                break

            # Predict the frame
            predict = model.predict(frame, conf=0.75)
            # Plot the prediction boxes on the frame
            plotted = predict[0].plot()

            # Display the video frame
            frame_placeholder.image(plotted, channels="BGR", caption="Video Frame")

            # Increase the current frame for smooth playback
            current_frame += 1
            if current_frame >= frame_count:  # Loop back to start
                current_frame = 0

        time.sleep(1 / fps)  # Control the playback speed based on the video's FPS
    
    cap.release()
    # Clean up the temporary file
    os.unlink(temp_file.name)


if __name__ == '__main__':
    main()
