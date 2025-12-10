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
mp_drawing = mp.solutions.drawing_utils
SEQ_LEN = 20

def landmarks_to_features(landmarks):
    """Convierte los landmarks de MediaPipe en un vector de características normalizado."""
    if landmarks is None or len(landmarks) != 21: return None
    
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    
    # Normalizar por posición de la muñeca
    wrist = pts[0].copy()
    pts_rel = pts - wrist
    
    # Normalizar por tamaño de la mano (distancia promedio de los dedos a la muñeca)
    dists = [np.linalg.norm(pts[i] - wrist) for i in [5, 9, 13, 17]]
    hand_size = (np.mean(dists) + 1e-6)
    pts_rel /= hand_size
    
    return pts_rel.flatten()

class SequenceBuffer:
    """Almacena una secuencia de vectores de características de longitud fija."""
    def __init__(self, maxlen=SEQ_LEN):
        self.frames = deque(maxlen=maxlen)
    def add(self, feats):
        if feats is not None: self.frames.append(feats)
    def is_ready(self):
        return len(self.frames) == self.frames.maxlen
    def get_sequence(self):
        return np.concatenate(list(self.frames))

class SignLanguageProcessor:
    # --- CAMBIO CRITICO: Recibe 'shared_models' para no cargar la IA por cada usuario ---
    def __init__(self, shared_models):
        
        # Referencias a los modelos globales (Memoria Compartida de Solo Lectura)
        self.clf_static = shared_models.get('session')
        self.classes_static = shared_models.get('classes')
        self.scaler_static = shared_models.get('scaler')
        
        # --- Inicialización de MediaPipe ---
        self.hands = mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.4, 
            model_complexity=0 # Optimización para CPU
        )

        # --- Estado de la Lógica de Reconocimiento (PROPIO DE CADA USUARIO) ---
        self.buffer = SequenceBuffer()
        self.prev_feats = None
        self.motion_hist = deque(maxlen=15)
        self.current_sentence = []
        self.last_added_letter = None
        self.last_hand_seen_time = time.time()
        
        # --- Configuración de Tiempos ---
        self.SPACE_TIMEOUT = 1.0        
        self.HOLD_TIME = 1.5            
        self.POST_WRITE_COOLDOWN = 0.5 
        
        # --- COLA DE EMISIÓN ---
        self.output_queue = deque(maxlen=1) 
        
        # --- Estados Temporales ---
        self.stable_prediction = None    
        self.stable_start_time = None    
        self.last_write_time = 0         
        self.last_status_is_success = False 

        # --- Debugging ---
        self.frame_saved = False 
        self.latest_frame = None
        self.lock = Lock()

    def process_frame(self, frame):
        """Procesa un único frame de la cámara."""
        
        # --- PRIVACIDAD: Comentamos esto para no guardar fotos en disco ---
        # if not self.frame_saved:
        #     cv2.imwrite("debug_frame_raw.jpg", frame)
        #     self.frame_saved = True 
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_detected = bool(results.multi_hand_landmarks)
        predicted_letter = None
        conf = 0.0

        if hand_detected:
            # Opcional: Dibujar landmarks (Consume CPU, quitar si deseas más velocidad)
            # mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            
            current_feats = landmarks_to_features(results.multi_hand_landmarks[0].landmark)
            
            if self.clf_static and current_feats is not None:
                X = current_feats.reshape(1, -1)
                X_scaled = self.scaler_static.transform(X)
                predicted_letter, conf = self._predict_onnx(self.clf_static, X_scaled, self.classes_static)
                # print(f"[DEBUG] Predicción: {predicted_letter}, Confianza: {conf:.2f}")

        # Lógica de frase
        self._update_sentence_logic(hand_detected, predicted_letter)
        
        with self.lock:
            self.latest_frame = frame.copy()

    def _predict_onnx(self, session, data, classes):
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: data.astype(np.float32)}
        ort_outs = session.run(None, ort_inputs)
        
        proba = self._softmax(ort_outs[0][0])
        conf = np.max(proba)
        pred_index = np.argmax(proba)
        predicted_class = classes[pred_index]

        if conf >= 0.6:
            return predicted_class, conf
        return None, 0

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _update_sentence_logic(self, hand_detected, predicted_letter):
        current_time = time.time()
        
        # 1. Cooldown
        if self.last_status_is_success:
            if (current_time - self.last_write_time) < self.POST_WRITE_COOLDOWN:
                return 
            else:
                self.last_status_is_success = False

        # 2. Mano Detectada
        if hand_detected and predicted_letter:
            self.last_hand_seen_time = current_time
            
            if predicted_letter != self.stable_prediction:
                self.stable_prediction = predicted_letter
                self.stable_start_time = current_time
            
            else:
                elapsed_stable_time = current_time - self.stable_start_time
                if elapsed_stable_time >= self.HOLD_TIME:
                    cooldown_ok = (current_time - self.last_write_time) > self.POST_WRITE_COOLDOWN
                    is_new_or_after_space = (
                        not self.current_sentence or 
                        self.current_sentence[-1].upper() != predicted_letter.upper() or 
                        self.current_sentence[-1] == ' '
                    )

                    if is_new_or_after_space and cooldown_ok:
                        self.current_sentence.append(predicted_letter.upper())
                        self.last_write_time = current_time
                        self.last_added_letter = predicted_letter.upper()
                        self.last_status_is_success = True 
                        self.stable_prediction = None 
                        self.stable_start_time = None
                        self._queue_update("".join(self.current_sentence))
        
        # 3. Mano NO Detectada (Espacio)
        if not hand_detected:
            self.stable_prediction = None
            self.stable_start_time = None
            
            if (time.time() - self.last_hand_seen_time) > self.SPACE_TIMEOUT:
                cooldown_ok = (time.time() - self.last_write_time) > self.POST_WRITE_COOLDOWN

                if self.current_sentence and self.current_sentence[-1] != ' ' and cooldown_ok:
                    self.current_sentence.append(' ')
                    self.last_added_letter = ' '
                    self.last_write_time = time.time()
                    self.last_status_is_success = False 
                    self._queue_update("".join(self.current_sentence))
                
                self.last_hand_seen_time = time.time()

    def _queue_update(self, text):
        # Solo encola si el texto cambió
        if not self.output_queue or self.output_queue[0] != text:
            self.output_queue.append(text)

    def clear_sentence(self):
        self.current_sentence = []
        self.last_added_letter = None
        self.last_write_time = time.time() 
        self.last_status_is_success = False
        self.stable_prediction = None
        self.stable_start_time = None
        self._queue_update("")