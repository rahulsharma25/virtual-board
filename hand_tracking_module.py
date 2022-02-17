import cv2 as cv
import mediapipe as mp
import time

class Hand_detector():
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode=mode
        self.max_hands=max_hands
        self.model_complexity=model_complexity
        self.detection_confidence=detection_confidence
        self.tracking_confidence=tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_img)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for i, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lm_list.append([i, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (0, 255, 255), -1)
        return lm_list

def main():
    detector = Hand_detector(detection_confidence=0.9)
    cap = cv.VideoCapture(0)

    p_time = 0
    c_time = 0

    while True:
        isTrue, frame = cap.read()
        img = detector.find_hands(frame)
        lm_list = detector.find_position(frame, draw=False)
        # if len(lm_list) != 0:
        #     print(lm_list[8])

        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        cv.putText(img, str(int(fps)), (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

        cv.imshow("Video", img)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

    return

if __name__ == "__main__":
    main()