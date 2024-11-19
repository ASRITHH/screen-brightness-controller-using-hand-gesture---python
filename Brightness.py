# Importing required libraries
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

# Initialize Mediapipe Hand Detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)
draw_utils = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(frame_rgb)

    # List to store landmark coordinates
    landmark_list = []

    # If hands are detected, process landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape  # Get dimensions of the frame
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append((idx, x, y))

            # Draw the detected landmarks on the frame
            draw_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Adjust screen brightness based on thumb and index finger distance
    if landmark_list:
        # Coordinates of thumb tip and index finger tip
        thumb_x, thumb_y = landmark_list[4][1], landmark_list[4][2]
        index_x, index_y = landmark_list[8][1], landmark_list[8][2]

        # Draw circles on the thumb and index finger tips
        cv2.circle(frame, (thumb_x, thumb_y), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (index_x, index_y), 7, (0, 255, 0), cv2.FILLED)

        # Draw a line between the thumb and index finger tips
        cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)

        # Calculate the distance between the thumb and index finger tips
        distance = hypot(index_x - thumb_x, index_y - thumb_y)

        # Map the distance to a brightness range (0 to 100)
        brightness_level = np.interp(distance, [15, 220], [0, 100])

        # Set the screen brightness
        sbc.set_brightness(int(brightness_level))

    # Display the processed video frame
    cv2.imshow('Hand Brightness Control', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
