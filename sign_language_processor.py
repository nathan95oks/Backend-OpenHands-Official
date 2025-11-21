import cv2
import time
import numpy as np
import onnxruntime
import joblib
from threading import Lock
from pathlib import Path
from collections import deque
import mediapipe as mp
from flask_socketio import SocketIO # Se añade para que la clase sepa de dónde llamar start_background_task

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

def motion_score(prev_feats, curr_feats):
    """Calcula la diferencia (norma euclidiana) entre dos vectores de características."""
    if prev_feats is None or curr_feats is None: return 0.0
    return np.linalg.norm(curr_feats - prev_feats)

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
# --------------------------------------------------------------------


class SignLanguageProcessor:
    def __init__(self, socketio):
        # --- Configuración ---
        self.socketio = socketio
        self.MODEL_STATIC_PATH = Path("models_onnx/lsb_alpha.onnx")
        self.MODEL_SEQ_PATH = Path("models_onnx/lsb_seq.onnx")
        
        # --- Carga de Modelos y Componentes ---
        self.clf_static, self.classes_static, self.scaler_static = self._load_onnx_model(self.MODEL_STATIC_PATH)
        self.clf_seq, self.classes_seq, self.scaler_seq = self._load_onnx_model(self.MODEL_SEQ_PATH)
        
        # --- Inicialización de MediaPipe ---
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

        # --- Estado de la Lógica de Reconocimiento ---
        self.buffer = SequenceBuffer()
        self.prev_feats = None
        self.motion_hist = deque(maxlen=15)
        self.current_sentence = []
        self.last_added_letter = None
        self.last_hand_seen_time = time.time()
        
        # --- Configuración de Tiempos Solicitados ---
        self.SPACE_TIMEOUT = 1.0        # Tiempo de pausa para añadir un espacio
        self.HOLD_TIME = 2.5            # <--- TIEMPO DE CAPTURA (2.5 segundos)
        self.POST_WRITE_COOLDOWN = 1.0  # Cooldown después de escribir para mostrar "CAPTURA COMPLETA"
        
        # --- NUEVOS ESTADOS para el Temporizador y el Feedback ---
        self.stable_prediction = None    # La letra que se ha estado detectando consistentemente
        self.stable_start_time = None    # El momento en que stable_prediction apareció por primera vez
        self.last_write_time = 0         # El momento en que se escribió la última letra/espacio
        self.last_status_is_success = False # Flag para mostrar el texto en verde
        
        # --- Frame para la App Web ---
        self.latest_frame = None
        self.lock = Lock()

    def _load_onnx_model(self, model_path):
        """Carga un modelo ONNX, su scaler y sus clases."""
        if not model_path.exists():
            print(f"[ERROR] Modelo no encontrado en {model_path}. Ejecuta train_pytorch.py primero.")
            return None, None, None
        
        try:
            session = onnxruntime.InferenceSession(str(model_path))
            classes = np.load(model_path.with_suffix('.classes.npy'))
            scaler = joblib.load(model_path.with_suffix('.scaler.joblib'))
            print(f"[INFO] Modelo ONNX cargado correctamente desde {model_path}")
            return session, classes, scaler
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo o sus componentes desde {model_path}: {e}")
            return None, None, None

    def process_frame(self, frame):
        """Procesa un único frame de la cámara."""
        frame = cv2.flip(frame, 1)
        
        # Procesamiento con MediaPipe en todo el fotograma
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_detected = bool(results.multi_hand_landmarks)
        predicted_letter = None

        if hand_detected:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            current_feats = landmarks_to_features(results.multi_hand_landmarks[0].landmark)
            
            if self.prev_feats is not None:
                score = motion_score(self.prev_feats, current_feats)
                self.motion_hist.append(score)
            self.prev_feats = current_feats

            moving = sum(s > 0.15 for s in self.motion_hist) >= 8
            
            if moving and self.clf_seq:
                self.buffer.add(current_feats)
                if self.buffer.is_ready():
                    X = self.buffer.get_sequence().reshape(1, -1)
                    X_scaled = self.scaler_seq.transform(X)
                    predicted_letter, _ = self._predict_onnx(self.clf_seq, X_scaled, self.classes_seq)
            elif not moving and self.clf_static:
                X = current_feats.reshape(1, -1)
                X_scaled = self.scaler_static.transform(X)
                predicted_letter, _ = self._predict_onnx(self.clf_static, X_scaled, self.classes_static)

        # La lógica de estado devuelve el texto Y el color
        status_text, text_color = self._update_sentence_logic(hand_detected, predicted_letter)
        
        # El color es BGR: Amarillo=(255, 255, 0), Verde=(0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        with self.lock:
            self.latest_frame = frame.copy()

    def _predict_onnx(self, session, data, classes):
        """Realiza una predicción usando la sesión de ONNX Runtime."""
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: data.astype(np.float32)}
        ort_outs = session.run(None, ort_inputs)
        
        proba = self._softmax(ort_outs[0][0])
        conf = np.max(proba)
        pred_index = np.argmax(proba)
        predicted_class = classes[pred_index]

        # Umbral de confianza
        if conf >= 0.6:
            return predicted_class, conf
        
        return None, 0

    def _softmax(self, x):
        """Calcula softmax para un array."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _update_sentence_logic(self, hand_detected, predicted_letter):
        """Gestiona el estado para añadir letras y espacios - LÓGICA CON CAPTURA DE 2.5s."""
        current_time = time.time()
        
        # Colores: Amarillo por defecto
        text_color = (255, 255, 0) 
        
        # 1. Lógica de Cooldown después de la última escritura
        if self.last_status_is_success:
            if (current_time - self.last_write_time) < self.POST_WRITE_COOLDOWN:
                # Mostrar el mensaje de éxito en VERDE por 1.0s
                return "¡CAPTURA COMPLETA! Seña agregada.", (0, 255, 0) # Verde
            else:
                self.last_status_is_success = False # El cooldown de éxito terminó

        # 2. Lógica de Mano Detectada y Captura
        if hand_detected and predicted_letter:
            self.last_hand_seen_time = current_time
            
            # --- MANEJO DE LA ESTABILIDAD (2.5s) ---
            
            # A. Seña diferente o inicio de estabilidad
            if predicted_letter != self.stable_prediction:
                self.stable_prediction = predicted_letter
                self.stable_start_time = current_time
                return f"Capturando: {predicted_letter.upper()}... (0.0s)", text_color
            
            # B. Seña estable - Contador corriendo
            elapsed_stable_time = current_time - self.stable_start_time
            
            if elapsed_stable_time < self.HOLD_TIME:
                time_remaining = self.HOLD_TIME - elapsed_stable_time
                status_text = f"Capturando: {predicted_letter.upper()}... ({time_remaining:.1f}s restantes)"
                return status_text, text_color
            
            # C. Seña capturada (Tiempo cumplido)
            if elapsed_stable_time >= self.HOLD_TIME:
                
                cooldown_ok = (current_time - self.last_write_time) > self.POST_WRITE_COOLDOWN
                is_new_or_after_space = (
                    not self.current_sentence or 
                    self.current_sentence[-1].upper() != predicted_letter.upper() or 
                    self.current_sentence[-1] == ' '
                )

                if is_new_or_after_space and cooldown_ok:
                    # ESCRIBIR LA LETRA
                    letter_to_add = predicted_letter.upper()
                    self.current_sentence.append(letter_to_add)
                    self.last_write_time = current_time
                    self.last_added_letter = letter_to_add
                    self.last_status_is_success = True # Activar el mensaje de éxito
                    
                    # Forzar el re-holding reseteando la predicción estable
                    self.stable_prediction = None 
                    self.stable_start_time = None
                    
                    self.socketio.start_background_task(
                        self._emit_update_async, "".join(self.current_sentence)
                    )
                    # Devuelve el mensaje de éxito inmediatamente
                    return "¡CAPTURA COMPLETA! Seña agregada.", (0, 255, 0) # Verde
                
                else:
                    # Seña cumplió el tiempo pero es la misma que la última y no ha habido cooldown/espacio
                    return f"Mantener {predicted_letter.upper()}. Cooldown/Cambio necesario.", text_color
        
        # 3. Lógica de Mano NO Detectada (o predicción baja)
        
        # Resetear el estado de estabilidad
        if not hand_detected:
            self.stable_prediction = None
            self.stable_start_time = None
            
            # Lógica de espacio: Añadir ' ' después de un periodo sin mano
            if (current_time - self.last_hand_seen_time) > self.SPACE_TIMEOUT:
                cooldown_ok = (current_time - self.last_write_time) > self.POST_WRITE_COOLDOWN

                if self.current_sentence and self.current_sentence[-1] != ' ' and cooldown_ok:
                    self.current_sentence.append(' ')
                    self.last_added_letter = ' '
                    self.last_write_time = current_time # Actualizar tiempo de escritura (del espacio)
                    self.last_status_is_success = False # El espacio no es una "captura de seña"
                    
                    self.socketio.start_background_task(
                        self._emit_update_async, "".join(self.current_sentence)
                    )
                    return "Espacio añadido. Muestre una seña.", text_color
                
                self.last_hand_seen_time = current_time # Resetear el temporizador de espacio

        # 4. Estado por defecto (No hay mano O mano con baja confianza O tiempo de espacio sin cumplirse)
        return "Muestre una seña a la camara.", text_color

    def _emit_update_async(self, text):
        """Método auxiliar para emitir actualizaciones de SocketIO de forma segura (asíncrona)."""
        self.socketio.emit('update_text', {'text': text})

    def clear_sentence(self):
        """Limpia la frase actual."""
        self.current_sentence = []
        self.last_added_letter = None
        self.last_write_time = time.time() # Resetear el cooldown al limpiar
        self.last_status_is_success = False
        self.stable_prediction = None
        self.stable_start_time = None
        
        self.socketio.start_background_task(
            self._emit_update_async, ""
        )
        print('[INFO] Texto borrado por el cliente.')