from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import time
from threading import Thread
from sign_language_processor import SignLanguageProcessor # <-- IMPORTANTE

# --- Configuración de la App ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Instancia única del procesador ---
# Se pasa socketio para que el procesador pueda emitir mensajes directamente
processor = SignLanguageProcessor(socketio)

# --- Hilo de Fondo para el Procesamiento de CV ---
def video_processing_thread():
    """Bucle principal que captura de la cámara y procesa los frames."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Iniciando hilo de procesamiento de video...")
    while True:
        success, frame = cap.read()
        if not success:
            socketio.sleep(0.01)
            continue
        
        # Toda la lógica compleja está ahora en este método
        processor.process_frame(frame)
        
        socketio.sleep(0.02) # Controla el framerate para no saturar la CPU

# --- Rutas de Flask y Eventos de Socket.IO ---
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    """Genera el stream de video para la página web."""
    while True:
        with processor.lock:
            if processor.latest_frame is None:
                continue
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

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')
    # Envía el estado actual de la frase al nuevo cliente
    socketio.emit('update_text', {'text': "".join(processor.current_sentence)})

@socketio.on('clear_text')
def handle_clear_text():
    """Manejador para el evento de borrar texto desde el cliente."""
    processor.clear_sentence()

if __name__ == '__main__':
    print("Iniciando la aplicación de traductor de señas...")
    # Iniciar el hilo de procesamiento de video en segundo plano
    Thread(target=video_processing_thread, daemon=True).start()
    # Ejecutar la aplicación web
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)