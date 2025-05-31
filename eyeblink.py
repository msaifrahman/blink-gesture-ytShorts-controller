import cv2
import mediapipe as mp
import pyautogui
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

class IndexFingerSwipeTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
        self.last_pos = None  # (x, y)
        self.swipe_threshold_x = 0.05  # horizontal swipe sensitivity
        self.swipe_threshold_y = 0.05  # vertical swipe sensitivity
        self.cooldown = 0.3  # seconds between triggers
        self.last_trigger = time.time()

    def detect_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        gesture = None
        hand_landmarks = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = hand_landmarks.landmark[8]
            current_pos = (index_tip.x, index_tip.y)

            if self.last_pos:
                dx = current_pos[0] - self.last_pos[0]
                dy = current_pos[1] - self.last_pos[1]

                # Invert horizontal swipe direction here
                if abs(dx) > self.swipe_threshold_x and abs(dx) > abs(dy):
                    gesture = "left" if dx > 0 else "right"  # inverted

                elif abs(dy) > self.swipe_threshold_y and abs(dy) > abs(dx):
                    gesture = "down" if dy > 0 else "up"

            self.last_pos = current_pos

        else:
            self.last_pos = None  # reset if no hand detected

        return gesture, hand_landmarks

class BlinkDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.blink_count = 0
        self.last_blink_time = 0
        self.blink_cooldown = 0.25  # seconds
        self.double_blink_window = 0.7  # seconds

    def detect_blink(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        gesture = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # Use left eye landmarks (33, 159, 145, 133, 153, 154, 155, 133)
            # We'll use 33 (left eye outer), 159 (top), 145 (bottom)
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            left_eye_outer = landmarks[33]
            left_eye_inner = landmarks[133]

            # Eye aspect ratio (vertical/horizontal)
            vert_dist = abs(left_eye_top.y - left_eye_bottom.y)
            horiz_dist = abs(left_eye_outer.x - left_eye_inner.x)
            ear = vert_dist / horiz_dist if horiz_dist != 0 else 0

            # Blink detected if EAR is below threshold
            if ear < 0.20:
                now = time.time()
                if now - self.last_blink_time > self.blink_cooldown:
                    self.blink_count += 1
                    self.last_blink_time = now

            # Check for gesture
            if self.blink_count == 1 and (time.time() - self.last_blink_time) > self.double_blink_window:
                gesture = "down"
                self.blink_count = 0
            elif self.blink_count == 2:
                gesture = "up"
                self.blink_count = 0

        else:
            # Reset if no face
            self.blink_count = 0

        return gesture

# ...existing code...
def trigger_key(gesture, tracker):
    now = time.time()
    if gesture and (now - tracker.last_trigger > tracker.cooldown):
        if gesture == "left":
            # Swipe right to left
            os.system('adb shell input swipe 800 800 200 800 200')
        elif gesture == "right":
            # Swipe left to right
            os.system('adb shell input swipe 200 800 800 800 200')
        elif gesture == "up":
            # Swipe bottom to top
            os.system('adb shell input swipe 500 1000 500 400 200')
        elif gesture == "down":
            # Swipe top to bottom
            os.system('adb shell input swipe 500 400 500 1000 200')
        tracker.last_trigger = now
# ...existing code...

def main():
    cap = cv2.VideoCapture(0)
    tracker = IndexFingerSwipeTracker()
    blink_detector = BlinkDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gesture, landmarks = tracker.detect_gesture(frame)
        blink_gesture = blink_detector.detect_blink(frame)

        if landmarks:
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # Use blink gestures for up/down, hand for left/right
        if blink_gesture in ["up", "down"]:
            cv2.putText(frame, f"BLINK: {blink_gesture.upper()}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            trigger_key(blink_gesture, tracker)
        elif gesture in ["left", "right"]:
            cv2.putText(frame, gesture.upper(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            trigger_key(gesture, tracker)

        cv2.imshow("Index Finger Swipe & Blink Controls", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()