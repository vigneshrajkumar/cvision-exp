import cv2
import time

from hand_tracking_module import HandDetector 

def main():

    cap = cv2.VideoCapture(0)

    prev_time = 0
    curr_time = 0

    detector = HandDetector()

    while True:
        success, frame = cap.read()
        frame = detector.find_hands(frame)
        landmarks = detector.find_position(frame)
        # this will track and print the pos of LM 0; wrist
        if len(landmarks) != 0:
            print(landmarks[0])
        
        # FPS computation & printing
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("image", frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()