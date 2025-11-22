from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import time
import base64
import numpy as np
import io
from collections import deque 
from sign_language_processor import SignLanguageProcessor 

# --- CONFIGURACIÓN DE MUESTREO DE FRAMES ---
FRAME_SKIP = 3  # Procesar 1 de cada 3 frames (reduce la carga y el ruido)
frame_counter = 0

# --- Configuración de la App ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Permitimos CORS para la app móvil/ngrok
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Instancia única del procesador ---
processor = SignLanguageProcessor(socketio)

# --- HILO DE EMISIÓN DE SOCKETIO SEGURO ---
def socket_emitter_thread():
    """Bucle que lee la cola del procesador y emite los mensajes de forma segura."""
    print("Iniciando hilo de emisión segura de SocketIO...")
    while True:
        # Solo emitimos si hay algo nuevo en la cola
        if processor.output_queue:
            text = processor.output_queue.popleft() # Saca el mensaje
            # Emitimos el mensaje directamente (es seguro en este hilo)
            socketio.emit('update_text', {'text': text})
            
        # Dormimos un momento para no saturar la CPU y permitir otros greenlets
        socketio.sleep(0.05) 

# --- Rutas de Flask y Eventos de Socket.IO ---
@app.route('/')
def index():
    return "Servidor de Reconocimiento de Señas LSB Activo. Use el cliente móvil."

def generate_frames():
    """Genera el stream de video para la página web (muestra el último frame procesado)."""
    while True:
        with processor.lock:
            if processor.latest_frame is None:
                frame_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame_placeholder, "Esperando frames del movil...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame_placeholder)
            else:
                ret, buffer = cv2.imencode('.jpg', processor.latest_frame)
            
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        socketio.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('video_frame')
def handle_mobile_frame(data):
    """Recibe un frame de la cámara móvil y lo pasa al procesador."""
    global frame_counter, FRAME_SKIP 
    
    frame_counter += 1
    if frame_counter % FRAME_SKIP != 0:
        return # Salta el frame para reducir la carga de procesamiento
        
    try:
        img_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            print(f"[DEBUG] Frame recibido y decodificado. Shape: {frame.shape}") 
            # El procesamiento (MediaPipe/ONNX) ocurre aquí.
            processor.process_frame(frame)
        
    except Exception as e:
        # print(f"[ERROR] No se pudo procesar el frame del móvil: {e}")
        pass

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')
    # Enviar el estado actual de la frase al nuevo cliente
    processor._queue_update("".join(processor.current_sentence))

@socketio.on('clear_text')
def handle_clear_text():
    """Manejador para el evento de borrar texto desde el cliente."""
    processor.clear_sentence()

if __name__ == '__main__':
    print("Iniciando el servidor de reconocimiento de señas...")
    
    # Iniciar el hilo de emisión de SocketIO que lee la cola
    socketio.start_background_task(target=socket_emitter_thread)
    
    # Ejecutar la aplicación web
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)