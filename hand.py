import mediapipe as mp
import cv2
import time


def hand_drawing():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands()

    while True:
        success, image = cap.read()

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(image, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        

        cv2.imshow("Hand Tracking", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
