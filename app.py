from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
import cv2
import time
import base64
import numpy as np
import onnxruntime
import joblib
from pathlib import Path
from collections import deque 
from sign_language_processor import SignLanguageProcessor 

# --- CONFIGURACIÓN ---
FRAME_SKIP = 2
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Permitimos CORS para la app móvil
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# ==============================================================================
# 1. CARGA GLOBAL DE MODELOS (CEREBRO COMPARTIDO)
# ==============================================================================
print("[SISTEMA] Cargando modelos ONNX en memoria global...")
MODEL_PATH = Path("models_onnx/lsb_alpha.onnx")
shared_models = {}

try:
    if MODEL_PATH.exists():
        shared_models['session'] = onnxruntime.InferenceSession(str(MODEL_PATH))
        shared_models['classes'] = np.load(MODEL_PATH.with_suffix('.classes.npy'))
        shared_models['scaler'] = joblib.load(MODEL_PATH.with_suffix('.scaler.joblib'))
        print("[SISTEMA] Modelos cargados EXITOSAMENTE.")
    else:
        print(f"[ERROR CRITICO] No se encuentra {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Falló la carga de modelos: {e}")

# ==============================================================================
# 2. GESTIÓN DE SESIONES (MEMORIA AISLADA)
# ==============================================================================
# Diccionario: { 'ID_USUARIO': Objeto_SignLanguageProcessor }
user_sessions = {}

# --- HILO DE EMISIÓN MULTI-USUARIO ---
def socket_emitter_thread():
    """Revisa la cola de cada usuario conectado y le envía SU mensaje privado."""
    print("[SISTEMA] Iniciando hilo de emisión multi-usuario...", flush=True)
    while True:
        # Usamos list(...) para evitar error si alguien se desconecta en el loop
        active_sids = list(user_sessions.keys())
        
        for sid in active_sids:
            try:
                processor = user_sessions.get(sid)
                if processor and processor.output_queue:
                    # Sacamos el mensaje de la cola de ESTE usuario
                    text = processor.output_queue.popleft()
                    # ENVIAMOS SOLO A ESTE USUARIO (Privacidad)
                    socketio.emit('update_text', {'text': text}, room=sid)
                    # print(f"[EMISIÓN] Enviando '{text}' a {sid}", flush=True)
            except Exception:
                continue
            
        socketio.sleep(0.05) 

# Iniciamos el hilo AL CARGAR EL ARCHIVO (Vital para Gunicorn)
socketio.start_background_task(target=socket_emitter_thread)


# --- RUTAS Y EVENTOS ---

@app.route('/')
def index():
    return f"Servidor LSB Multi-Usuario Activo. Conectados: {len(user_sessions)}"

@app.route('/video_feed')
def video_feed():
    # Nota: El video feed web solo mostrará 'Esperando' ya que los frames ahora son privados
    # Esto ahorra recursos en el servidor.
    return "Video Feed desactivado para ahorrar recursos en modo multi-usuario."

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"[CONEXION] Usuario conectado: {sid}", flush=True)
    
    if not shared_models:
        print("[ERROR] Modelos no cargados, imposible procesar.")
        return

    # CREAMOS UN PROCESADOR NUEVO SOLO PARA ESTE USUARIO
    user_sessions[sid] = SignLanguageProcessor(shared_models)
    
    # Enviarle su estado inicial (vacío)
    socketio.emit('update_text', {'text': ""}, room=sid)

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in user_sessions:
        del user_sessions[sid] # LIBERAR MEMORIA RAM
    print(f"[DESCONEXION] Usuario {sid} desconectado.", flush=True)

@socketio.on('video_frame')
def handle_mobile_frame(data):
    sid = request.sid
    processor = user_sessions.get(sid)
    
    # Si no existe sesión (raro) o no hay modelos, ignorar
    if not processor: return

    try:
        # Procesar solo si el cliente envía imagen válida
        img_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            processor.process_frame(frame)
    except Exception:
        pass

@socketio.on('clear_text')
def handle_clear_text():
    sid = request.sid
    processor = user_sessions.get(sid)
    if processor:
        print(f"[ACCION] Usuario {sid} borró su texto.", flush=True)
        processor.clear_sentence()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)