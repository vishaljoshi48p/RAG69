import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

def detect_hand_gestures(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            h, w, _ = frame.shape
            for landmark in hand_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            # Extract landmarks for gesture detection
            landmarks = results.multi_hand_landmarks[0]
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Print gesture detection results (or handle them server-side)
            if thumb_tip.y < index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
                print("Right gesture detected")
            elif thumb_tip.y > index_tip.y > middle_tip.y > ring_tip.y > pinky_tip.y:
                print("Left gesture detected")

    return frame

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Detect hand gestures
        frame_result = detect_hand_gestures(frame)

        # Show results (remove or modify as needed)
        cv2.imshow('Gesture Control', frame_result)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
