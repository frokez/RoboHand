import cv2
import numpy as np
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.6, tracking_conf=0.6, mirror=True):
        self.mirror = mirror
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

    def process(self, frame_bgr):
        """
        frame_bgr: OpenCV frame (H,W,3) BGR
        returns: (landmarks_21x3 or None, annotated_frame_bgr)
        """
        if self.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        annot = frame_bgr.copy()
        lm2d = None
        lm3d = None
        if result.multi_hand_landmarks:
            # pick the first detected hand (you set max_hands=1 anyway)
            hand_lms = result.multi_hand_landmarks[0]
            pts = []
            for lm_i in hand_lms.landmark:
                x_px = lm_i.x * w
                y_px = lm_i.y * h
                z_n = lm_i.z  # relative depth (negative is closer); keep as-is
                pts.append((x_px, y_px, z_n))
            lm2d = np.array(pts, dtype=np.float32)
        if result.multi_hand_world_landmarks:
            wld = result.multi_hand_world_landmarks[0]
            pts3 = []
            for p in wld.landmark:
                pts3.append((p.x, p.y, p.z))  
            lm3d = np.array(pts3, dtype=np.float32)
        if result.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
            annot,
            result.multi_hand_landmarks[0],
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )
        return lm2d, lm3d, annot


    def close(self):
        self.hands.close()
