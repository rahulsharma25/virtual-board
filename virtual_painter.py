import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import hand_tracking_module as htm
import os

def main():
    folder_path = "V-board overlay images"
    overlay_images = []
    for path in os.listdir(folder_path):
        image = cv.imread(os.path.join(folder_path, path))
        overlay_images.append(image)

    header = overlay_images[0]

    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = htm.Hand_detector(detection_confidence=0.8)

    finger_tip_ids = [4, 8, 12, 16, 20]

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    px, py = 0, 0
    draw_color = (255, 0, 0)

    ERASER_SIZE = 15
    BRUSH_SIZE = 4

    brush_thickness = BRUSH_SIZE

    p_time = 0
    while True:
        isTrue, img = cap.read()
        img = cv.flip(img, 1)

        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        img[:100] = header

        if len(lm_list) != 0:
            fingers_open_status = []

            if lm_list[finger_tip_ids[0]][1] < lm_list[finger_tip_ids[0]-1][1]:
                fingers_open_status.append(1)
            else:
                fingers_open_status.append(0)
            
            for i in range(1, 5):
                if lm_list[finger_tip_ids[i]][2] < lm_list[finger_tip_ids[i]-1][2]:
                    fingers_open_status.append(1)
                else:
                    fingers_open_status.append(0)

            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]
            cv.circle(img, (x1, y1), 6, (255, 255, 0), -1)
            cv.circle(img, (x2, y2), 6, (255, 255, 0), -1)

            # Selection Mode
            if fingers_open_status[1] and fingers_open_status[2]:
                px, py = x1, y1
                if y1 < 100:
                    if x1 > 37 and x1 < 152:
                        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                    if x1 > 236 and x1 < 308:
                        draw_color = (0, 0, 255)
                    if x1 > 344 and x1 < 416:
                        draw_color = (255, 0, 0)
                    if x1 > 458 and x1 < 530:
                        draw_color = (0, 255, 0)
                    if x1 > 568 and x1 < 640:
                        draw_color = (0, 255, 255)
                    if x1 > 674 and x1 < 746:
                        draw_color = (203, 192, 255)
                    if x1 > 779 and x1 < 847:
                        draw_color = (0, 0, 0)
                    if x1 > 884 and x1 < 902:
                        BRUSH_SIZE = 4
                        brush_thickness = BRUSH_SIZE
                    if x1 > 943 and x1 < 960:
                        BRUSH_SIZE = 8
                        brush_thickness = BRUSH_SIZE
                    if x1 > 987 and x1 < 1023:
                        BRUSH_SIZE = 12
                        brush_thickness = BRUSH_SIZE
                    if x1 > 1055 and x1 < 1090:
                        ERASER_SIZE = 15
                        brush_thickness = ERASER_SIZE
                    if x1 > 1112 and x1 < 1147:
                        ERASER_SIZE = 30
                        brush_thickness = ERASER_SIZE
                    if x1 > 1173 and x1 < 1208:
                        ERASER_SIZE = 50
                        brush_thickness = ERASER_SIZE

            # Drawing Mode
            if fingers_open_status[1] and (fingers_open_status[2] == 0):
                if px == 0 and py == 0:
                    px, py = x1, y1
                canvas = cv.line(canvas, (px, py), (x1, y1), draw_color, brush_thickness)
                px, py = x1, y1

            #Do Nothing
            if fingers_open_status[1] == False:
                px, py = 0, 0

        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        cv.putText(img, f'FPS: {int(fps)}', (10, 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        gray_img = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
        
        _, inverse_img = cv.threshold(gray_img, 1, 255, cv.THRESH_BINARY_INV)
        inverse_img = cv.cvtColor(inverse_img, cv.COLOR_GRAY2BGR)
        img = cv.bitwise_and(img, inverse_img)
        img = cv.bitwise_or(img, canvas)

        cv.imshow("Video", img)
        # cv.imshow("Canvas", canvas)
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()