import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("V_1.mp4")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        #continue

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (64, 64))  # Adjust size according to your model input
    #normalized_frame = resized_frame / 255.0
    #input_frame = np.expand_dims(normalized_frame, axis=0)

    # Show frame (showing resized_frame here instead of input_frame for compatibility with OpenCV)
    #cv2.imshow('Violence Detection', resized_frame)

    cv2.imshow('Violence Detection', resized_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()