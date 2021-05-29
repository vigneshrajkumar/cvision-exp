import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode = False, max_hands = 2, det_conf = 0.5, track_conf = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.det_conf = det_conf
        self.track_conf = track_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.det_conf, self.track_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw = True):
        # converts the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # processing the rgb frame
        self.results = self.hands.process(image_rgb)
        
        # drawing a skeletion for every palm detected
        if self.results.multi_hand_landmarks:
            for hand_lm in self.results.multi_hand_landmarks:    
                if draw:               
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        return frame
    
    def find_position(self, frame, hand_no = 0, draw = False):
        landmarks = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                height, width, center = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                landmarks.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
                
        return landmarks