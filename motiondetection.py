import cv2
import numpy as np
import pyaudio
import threading
import winsound  # For sound alert on Windows
import time

# Constants for audio detection
SILENCE_THRESHOLD = 500  # Adjust this value for ambient noise
SOUND_THRESHOLD = 5000   # Threshold for detecting sound (e.g., speech)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Initialize the microphone
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Store previous center positions for motion tracking
previous_positions = []

# Initialize HOG descriptor and SVM for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize global variables for sound alert timing
last_sound_time = time.time()
sound_interval = 3  # Sound alert interval in seconds

def sound_detection():
    global last_sound_time  # Declare as global at the beginning of the function
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        avg_volume = np.abs(data).mean()

        # Check if the average volume exceeds the silence threshold and the sound threshold
        if avg_volume > SILENCE_THRESHOLD and avg_volume > SOUND_THRESHOLD:
            current_time = time.time()
            if current_time - last_sound_time > sound_interval:  # Check interval
                print("Sound Detected!")
                winsound.Beep(1000, 500)  # Sound alert (frequency, duration)
                last_sound_time = current_time  # Update last sound time

def motion_detection():
    global last_sound_time  # Declare as global in this function
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Detect humans
        boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8, 8), padding=(8, 8), scale=1.05)

        # Draw rectangles around detected humans
        for (x, y, w, h) in boxes:
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Store position for motion path
            previous_positions.append((center_x, center_y))
            if len(previous_positions) > 1:
                for i in range(len(previous_positions) - 1):
                    cv2.line(frame_resized, previous_positions[i], previous_positions[i + 1], (0, 0, 255), 2)

        if len(boxes) > 0:
            cv2.putText(frame_resized, "Human Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            current_time = time.time()
            if current_time - last_sound_time > sound_interval:  # Check interval
                winsound.Beep(1000, 500)  # Sound alert for human detection
                last_sound_time = current_time  # Update last sound time

        cv2.imshow("Human Detection", frame_resized)

        if cv2.waitKey(10) == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Start threads for sound and motion detection
threading.Thread(target=sound_detection, daemon=True).start()
motion_detection()
