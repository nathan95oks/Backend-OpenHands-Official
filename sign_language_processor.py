import cv2
import time
import numpy as np
import onnxruntime
import joblib
from threading import Lock
from pathlib import Path
from collections import deque
import mediapipe as mp

# --- DEFINICIONES INTEGRADAS ---
mp_hands = mp.solutions.hands
SEQ_LEN = 20

def landmarks_to_features(landmarks):
    if landmarks is None or len(landmarks) != 21: return None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts_rel = pts - wrist
    dists = [np.linalg.norm(pts[i] - wrist) for i in [5, 9, 13, 17]]
    hand_size = (np.mean(dists) + 1e-6)
    pts_rel /= hand_size
    return pts_rel.flatten()

class SequenceBuffer:
    def __init__(self, maxlen=SEQ_LEN):
        self.frames = deque(maxlen=maxlen)
    def add(self, feats):
        if feats is not None: self.frames.append(feats)
    def is_ready(self):
        return len(self.frames) == self.frames.maxlen
    def get_sequence(self):
        return np.concatenate(list(self.frames))

class SignLanguageProcessor:
    def __init__(self, shared_models):
        self.clf_static = shared_models.get('session')
        self.classes_static = shared_models.get('classes')
        self.scaler_static = shared_models.get('scaler')
        
        # --- OPTIMIZACIÓN 1: Configuración ultrarrápida de MediaPipe ---
        # model_complexity=0 (Lite) y min_tracking_confidence bajo para que no pierda el rastro fácil
        self.hands = mp_hands.Hands(
            static_image_mode=False, # Importante: False activa el tracking (más rápido que detectar)
            max_num_hands=1, 
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            model_complexity=0 
        )

        self.current_sentence = []
        self.output_queue = deque(maxlen=1) 
        
        # --- OPTIMIZACIÓN 2: Salto de Frames Agresivo ---
        # Procesar 1 frame, saltar 2. Reduce la carga del CPU al 33%.
        self.frame_counter = 0
        self.FRAME_SKIP = 3 

        self.stable_prediction = None    
        self.stable_start_time = None    
        self.last_write_time = 0         
        self.last_status_is_success = False 
        
        # Tiempos
        self.SPACE_TIMEOUT = 1.0        
        self.HOLD_TIME = 1.5            
        self.POST_WRITE_COOLDOWN = 0.5 
        
        self.last_hand_seen_time = time.time()
        self.latest_frame = None
        self.lock = Lock()

    def process_frame(self, frame):
        # 1. Skip Frames: Si llega mucho video, ignoramos la mayoría para no saturar CPU
        self.frame_counter += 1
        if self.frame_counter % self.FRAME_SKIP != 0:
            return

        # --- OPTIMIZACIÓN 3: REDIMENSIONAR (La clave de la velocidad) ---
        # Reducimos la imagen a una altura de 240px o 256px.
        # MediaPipe no necesita HD. Procesar 240p es 4 veces más rápido que 480p.
        h, w, _ = frame.shape
        scale_factor = 240 / h
        new_w = int(w * scale_factor)
        small_frame = cv2.resize(frame, (new_w, 240))

        # Convertir a RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen pequeña
        results = self.hands.process(rgb_frame)
        
        hand_detected = bool(results.multi_hand_landmarks)
        predicted_letter = None

        if hand_detected:
            current_feats = landmarks_to_features(results.multi_hand_landmarks[0].landmark)
            if self.clf_static and current_feats is not None:
                X = current_feats.reshape(1, -1)
                X_scaled = self.scaler_static.transform(X)
                predicted_letter, conf = self._predict_onnx(self.clf_static, X_scaled, self.classes_static)

        self._update_sentence_logic(hand_detected, predicted_letter)

    def _predict_onnx(self, session, data, classes):
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: data.astype(np.float32)}
        ort_outs = session.run(None, ort_inputs)
        
        x = ort_outs[0][0]
        e_x = np.exp(x - np.max(x))
        proba = e_x / e_x.sum(axis=0)
        
        conf = np.max(proba)
        predicted_class = classes[np.argmax(proba)]

        if conf >= 0.6: return predicted_class, conf
        return None, 0

    def _update_sentence_logic(self, hand_detected, predicted_letter):
        current_time = time.time()
        
        if self.last_status_is_success:
            if (current_time - self.last_write_time) < self.POST_WRITE_COOLDOWN: return 
            else: self.last_status_is_success = False

        if hand_detected and predicted_letter:
            self.last_hand_seen_time = current_time
            if predicted_letter != self.stable_prediction:
                self.stable_prediction = predicted_letter
                self.stable_start_time = current_time
            else:
                if (current_time - self.stable_start_time) >= self.HOLD_TIME:
                    cooldown_ok = (current_time - self.last_write_time) > self.POST_WRITE_COOLDOWN
                    is_new = (not self.current_sentence or 
                              self.current_sentence[-1].upper() != predicted_letter.upper() or 
                              self.current_sentence[-1] == ' ')

                    if is_new and cooldown_ok:
                        self.current_sentence.append(predicted_letter.upper())
                        self.last_write_time = current_time
                        self.last_status_is_success = True 
                        self.stable_prediction = None 
                        self._queue_update("".join(self.current_sentence))
        
        elif not hand_detected:
            self.stable_prediction = None
            if (time.time() - self.last_hand_seen_time) > self.SPACE_TIMEOUT:
                if self.current_sentence and self.current_sentence[-1] != ' ':
                    self.current_sentence.append(' ')
                    self.last_write_time = time.time()
                    self._queue_update("".join(self.current_sentence))
                self.last_hand_seen_time = time.time()

    def _queue_update(self, text):
        if not self.output_queue or self.output_queue[0] != text:
            self.output_queue.append(text)

    def clear_sentence(self):
        self.current_sentence = []
        self._queue_update("")