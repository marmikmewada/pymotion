import cv2
import numpy as np
import pyaudio
import threading
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

# Initialize global variable for last position
last_position = None

def sound_detection():
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        avg_volume = np.abs(data).mean()

        # Check if the average volume exceeds the silence threshold and the sound threshold
        if avg_volume > SILENCE_THRESHOLD and avg_volume > SOUND_THRESHOLD:
            print("Sound Detected!")

def motion_tracking():
    global last_position  # Declare as global to modify the outer variable
    cap = cv2.VideoCapture(0)

    # Background subtractor for motion detection
    back_sub = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Apply background subtraction
        fg_mask = back_sub.apply(frame_resized)

        # Find contours of the moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around moving objects and track their motion
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Minimum area to filter noise
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw rectangle around detected motion
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Store position for motion path
            current_position = (center_x, center_y)

            if last_position is not None:
                # Determine the direction of movement
                dx = current_position[0] - last_position[0]
                dy = current_position[1] - last_position[1]

                if abs(dx) > abs(dy):  # Horizontal movement
                    if dx > 0:
                        print("Moving Right")
                    else:
                        print("Moving Left")
                else:  # Vertical movement
                    if dy > 0:
                        print("Moving Down")
                    else:
                        print("Moving Up")

            last_position = current_position

        cv2.imshow("Motion Tracking", frame_resized)

        if cv2.waitKey(10) == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Start threads for sound detection
threading.Thread(target=sound_detection, daemon=True).start()
motion_tracking()
