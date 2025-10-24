from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import io
import threading
import time
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# Variáveis globais
picam2 = None
frame_lock = threading.Lock()
current_frame = None

# Métricas globais
metrics = {
    'frame_count': 0,
    'total_capture_time': 0,
    'total_encode_time': 0,
    'fps_real': 0,
    'avg_capture_ms': 0,
    'avg_encode_ms': 0,
    'start_time': None,
    'last_update': 0
}

TARGET_FPS = 24
RESOLUTION = (640, 480)

def initialize_camera():
    """Inicializa a câmera Picamera2"""
    global picam2

    try:
        picam2 = Picamera2()

        # Configuração otimizada para streaming
        config = picam2.create_video_configuration(
            main={"size": RESOLUTION, "format": "BGR888"},
            controls={
                'FrameRate': TARGET_FPS,
                'NoiseReductionMode': 0
            }
        )

        picam2.configure(config)
        picam2.start()

        print("Câmera Picamera2 inicializada com sucesso!")
        time.sleep(2)  # Aguarda estabilização
        return True

    except Exception as e:
        print(f"Erro ao inicializar câmera: {e}")
        return False

def capture_frames():
    """Thread que captura frames continuamente"""
    global current_frame, metrics

    metrics['start_time'] = time.time()

    while True:
        try:
            # Capturar frame
            capture_start = time.time()
            array = picam2.capture_array()
            capture_time = time.time() - capture_start

            # Converter para JPEG
            encode_start = time.time()
            # Converter RGB para BGR
            array_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            # Codificar em JPEG usando OpenCV
            success, buffer = cv2.imencode('.jpg', array_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            encode_time = time.time() - encode_start

            # Atualizar frame atual
            with frame_lock:
                current_frame = frame_bytes

            # Atualizar métricas
            metrics['frame_count'] += 1
            metrics['total_capture_time'] += capture_time
            metrics['total_encode_time'] += encode_time

            # Atualizar métricas calculadas a cada segundo
            current_time = time.time()
            if current_time - metrics['last_update'] >= 1.0:
                elapsed = current_time - metrics['start_time']
                metrics['fps_real'] = metrics['frame_count'] / elapsed
                metrics['avg_capture_ms'] = (metrics['total_capture_time'] / metrics['frame_count']) * 1000
                metrics['avg_encode_ms'] = (metrics['total_encode_time'] / metrics['frame_count']) * 1000
                metrics['last_update'] = current_time

            # Pequeno delay para controlar taxa de captura
            time.sleep(0.01)

        except Exception as e:
            print(f"Erro na captura: {e}")
            time.sleep(0.1)

def generate_frames():
    """Gerador de frames para streaming"""
    while True:
        with frame_lock:
            if current_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        time.sleep(0.04)  # ~25 fps no stream

@app.route('/')
def index():
    """Página principal com stream e métricas"""
    html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Pi Camera Stream - Métricas em Tempo Real</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }

        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }

        .content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            padding: 30px;
        }

        .video-container {
            position: relative;
        }

        .video-container img {
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .metrics-panel {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
        }

        .metrics-panel h2 {
            margin-top: 0;
            color: #667eea;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }

        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        }

        .metric-label {
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }

        .metric-unit {
            font-size: 0.6em;
            color: #999;
            font-weight: normal;
        }

        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }

        .status-good {
            background: #4caf50;
            color: white;
        }

        .status-warning {
            background: #ff9800;
            color: white;
        }

        .status-bad {
            background: #f44336;
            color: white;
        }

        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
        }

        .refresh-info {
            text-align: center;
            color: #999;
            font-size: 0.85em;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Raspberry Pi Camera Stream</h1>
            <p>Monitoramento de Desempenho em Tempo Real</p>
        </div>

        <div class="content">
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Camera Stream">
            </div>

            <div class="metrics-panel">
                <h2>📊 Métricas ao Vivo</h2>

                <div class="metric-card">
                    <div class="metric-label">FPS Real</div>
                    <div class="metric-value" id="fps">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Frames Capturados</div>
                    <div class="metric-value" id="frames">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Tempo de Captura</div>
                    <div class="metric-value">
                        <span id="capture">--</span> <span class="metric-unit">ms</span>
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Tempo de Encode</div>
                    <div class="metric-value">
                        <span id="encode">--</span> <span class="metric-unit">ms</span>
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Status do Sistema</div>
                    <div id="status" style="margin-top: 10px;">
                        <span class="status-badge status-good">Inicializando...</span>
                    </div>
                </div>

                <div class="refresh-info">
                    Atualização automática a cada segundo
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateMetrics() {
            fetch('/metrics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps_real.toFixed(2);
                    document.getElementById('frames').textContent = data.frame_count;
                    document.getElementById('capture').textContent = data.avg_capture_ms.toFixed(2);
                    document.getElementById('encode').textContent = data.avg_encode_ms.toFixed(2);

                    // Atualizar status
                    const statusDiv = document.getElementById('status');
                    let statusHTML = '';

                    if (data.fps_real >= 20) {
                        statusHTML = '<span class="status-badge status-good">✓ Excelente</span>';
                    } else if (data.fps_real >= 15) {
                        statusHTML = '<span class="status-badge status-warning">⚠ Bom</span>';
                    } else {
                        statusHTML = '<span class="status-badge status-bad">✗ Atenção</span>';
                    }

                    statusDiv.innerHTML = statusHTML;
                })
                .catch(error => console.error('Erro ao atualizar métricas:', error));
        }

        // Atualizar métricas a cada segundo
        setInterval(updateMetrics, 1000);

        // Primeira atualização imediata
        updateMetrics();
    </script>
</body>
</html>
    '''
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    """Endpoint de streaming de vídeo"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def get_metrics():
    """Endpoint JSON para métricas"""
    return {
        'frame_count': metrics['frame_count'],
        'fps_real': metrics['fps_real'],
        'avg_capture_ms': metrics['avg_capture_ms'],
        'avg_encode_ms': metrics['avg_encode_ms'],
        'total_time': time.time() - metrics['start_time'] if metrics['start_time'] else 0
    }

if __name__ == '__main__':
    # Inicializar câmera
    if not initialize_camera():
        print("Falha ao inicializar câmera. Encerrando...")
        exit(1)

    # Iniciar thread de captura
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    print("\n" + "="*60)
    print("Servidor Flask iniciado!")
    print("="*60)
    print("Acesse no navegador:")
    print("  - Local: http://localhost:5000")
    print("  - Rede:  http://<IP_DA_RASPBERRY>:5000")
    print("="*60)
    print("Pressione Ctrl+C para encerrar\n")

    # Iniciar servidor Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
