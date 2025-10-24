from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2
import io
import threading
import time
from PIL import Image
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from queue import Queue
import psutil
import os

app = Flask(__name__)

# Variáveis globais
picam2 = None
frame_lock = threading.Lock()
current_frame = None
annotated_frame = None
is_classifying = False

# Thresholds adaptativos por classe
CLASS_THRESHOLDS = {
    'background': 0.50,
    'car': 0.25,
    'sign': 0.35
}

model_path = "./models/ei-iesti05---trabalho-1-quantized.lite"
labels = ['background', 'car', 'sign']

MODEL_METRICS = {
    'car': {'precision': 0.849, 'recall': 0.743, 'f1': 0.792},
    'sign': {'precision': 0.686, 'recall': 0.712, 'f1': 0.699}
}

interpreter = None
classification_queue = Queue(maxsize=1)
inference_lock = threading.Lock()

# Cores BGR (OpenCV)
COLORS_BGR = {
    'background': (128, 128, 128),
    'car': (0, 255, 0),
    'sign': (0, 0, 255)
}

metrics = {
    'frame_count': 0,
    'total_capture_time': 0,
    'total_encode_time': 0,
    'total_inference_time': 0,
    'inference_count': 0,
    'fps_capture': 0,           # FPS de captura (sem inferência)
    'fps_with_inference': 0,    # FPS real com inferência
    'avg_capture_ms': 0,
    'avg_encode_ms': 0,
    'avg_inference_ms': 0,
    'avg_total_pipeline_ms': 0, # Tempo total captura+inferência
    'slack_ms': 0,              # Tempo disponível em relação a 1/24s
    'start_time': None,
    'last_update': 0,
    'cpu_percent': 0,
    'cpu_temp': 0,
    'total_detections': {'car': 0, 'sign': 0},
    'last_pipeline_time': 0     # Último tempo de pipeline completo
}

TARGET_FPS = 24
FRAME_BUDGET_MS = 1000.0 / TARGET_FPS  # 41.67ms para 24 FPS
RESOLUTION = (640, 480)

def get_cpu_temperature():
    try:
        temp = os.popen("vcgencmd measure_temp").readline()
        return float(temp.replace("temp=","").replace("'C","").strip())
    except:
        return 0.0

def initialize_camera():
    global picam2
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": RESOLUTION, "format": "RGB888"},
            controls={'FrameRate': TARGET_FPS, 'NoiseReductionMode': 0}
        )
        picam2.configure(config)
        picam2.start()
        print("✓ Câmera inicializada (BGR888)")
        time.sleep(2)
        return True
    except Exception as e:
        print(f"Erro ao inicializar câmera: {e}")
        return False

def load_model():
    global interpreter
    if interpreter is None:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"\n✓ Modelo carregado: {model_path}")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Output shape: {output_details[0]['shape']}")
    return interpreter

def get_confidence_color(base_color_bgr, confidence):
    factor = 0.5 + (confidence * 0.5)
    return tuple(int(c * factor) for c in base_color_bgr)

def draw_system_overlay(img, metrics_data):
    """
    Desenha overlay com todas as métricas - SEMPRE visível
    """
    overlay_height = 170
    overlay_width = 240
    margin = 10

    y_start = img.shape[0] - overlay_height - margin
    x_start = margin

    # Fundo semi-transparente
    overlay = img.copy()
    cv2.rectangle(overlay, 
                 (x_start, y_start), 
                 (x_start + overlay_width, y_start + overlay_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Borda
    cv2.rectangle(img, 
                 (x_start, y_start), 
                 (x_start + overlay_width, y_start + overlay_height),
                 (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 20
    text_x = x_start + 8
    text_y = y_start + 16

    # FPS de captura (sem inferência)
    fps_cap = metrics_data['fps_capture']
    fps_cap_color = (0, 255, 0) if fps_cap >= 20 else (0, 165, 255) if fps_cap >= 15 else (0, 0, 255)
    cv2.putText(img, f"FPS Cap: {fps_cap:.1f}", 
               (text_x, text_y), font, font_scale, fps_cap_color, thickness)

    # FPS com inferência (real)
    text_y += line_height
    fps_inf = metrics_data['fps_with_inference']
    fps_inf_color = (0, 255, 0) if fps_inf >= 20 else (0, 165, 255) if fps_inf >= 15 else (0, 0, 255)
    if is_classifying:
        cv2.putText(img, f"FPS+Inf: {fps_inf:.1f}", 
                   (text_x, text_y), font, font_scale, fps_inf_color, thickness)
    else:
        cv2.putText(img, f"FPS+Inf: N/A", 
                   (text_x, text_y), font, font_scale, (128, 128, 128), thickness)

    # Tempo total do pipeline
    text_y += line_height
    pipeline_ms = metrics_data['avg_total_pipeline_ms']
    pipeline_color = (0, 255, 0) if pipeline_ms < 35 else (0, 165, 255) if pipeline_ms < 41.67 else (0, 0, 255)
    if is_classifying:
        cv2.putText(img, f"Pipeline: {pipeline_ms:.1f}ms", 
                   (text_x, text_y), font, font_scale, pipeline_color, thickness)
    else:
        cv2.putText(img, f"Pipeline: N/A", 
                   (text_x, text_y), font, font_scale, (128, 128, 128), thickness)

    # Slack disponível
    text_y += line_height
    slack = metrics_data['slack_ms']
    slack_color = (0, 255, 0) if slack > 5 else (0, 165, 255) if slack > 0 else (0, 0, 255)
    if is_classifying:
        cv2.putText(img, f"Slack: {slack:.1f}ms", 
                   (text_x, text_y), font, font_scale, slack_color, thickness)
    else:
        cv2.putText(img, f"Slack: N/A", 
                   (text_x, text_y), font, font_scale, (128, 128, 128), thickness)

    # CPU
    text_y += line_height
    cpu = metrics_data['cpu_percent']
    cpu_color = (0, 255, 0) if cpu < 50 else (0, 165, 255) if cpu < 75 else (0, 0, 255)
    cv2.putText(img, f"CPU: {cpu:.1f}%", 
               (text_x, text_y), font, font_scale, cpu_color, thickness)

    # Temperatura
    text_y += line_height
    temp = metrics_data['cpu_temp']
    temp_color = (0, 255, 0) if temp < 65 else (0, 165, 255) if temp < 75 else (0, 0, 255)
    cv2.putText(img, f"Temp: {temp:.1f}C", 
               (text_x, text_y), font, font_scale, temp_color, thickness)

    # Contadores de detecção (apenas se classificando)
    if is_classifying:
        det_counts = metrics_data['total_detections']
        text_y += line_height
        cv2.putText(img, f"Cars: {det_counts['car']}", 
                   (text_x, text_y), font, font_scale, COLORS_BGR['car'], thickness)

        text_y += line_height
        cv2.putText(img, f"Signs: {det_counts['sign']}", 
                   (text_x, text_y), font, font_scale, COLORS_BGR['sign'], thickness)

    return img

def detect_objects_fomo(img, interpreter, original_bgr, metrics_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Converter BGR para RGB para PIL
    img_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    img_resized = img_pil.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    input_data = np.expand_dims(np.array(img_resized), axis=0).astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    output_dtype = output_details[0]['dtype']
    if output_dtype in [np.int8, np.uint8]:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    if output.ndim == 4:
        output = output[0]

    grid_h, grid_w, num_classes = output.shape

    annotated = original_bgr.copy()
    img_h, img_w = annotated.shape[:2]

    scale_x = img_w / grid_w
    scale_y = img_h / grid_h

    class_counts = np.zeros(num_classes)
    detections = []

    for y in range(grid_h):
        for x in range(grid_w):
            pixel_probs = output[y, x, :]
            max_class = np.argmax(pixel_probs)
            max_prob = pixel_probs[max_class]

            class_counts[max_class] += 1

            if max_class != 0:
                label_name = labels[max_class]
                threshold = CLASS_THRESHOLDS[label_name]

                if max_prob >= threshold:
                    center_x = int((x + 0.5) * scale_x)
                    center_y = int((y + 0.5) * scale_y)

                    base_color = COLORS_BGR.get(label_name, (255, 255, 255))
                    color = get_confidence_color(base_color, max_prob)

                    radius = int(5 + (max_prob - threshold) * 10)
                    cv2.circle(annotated, (center_x, center_y), radius, color, -1)

                    x1 = int(x * scale_x)
                    y1 = int(y * scale_y)
                    x2 = int((x + 1) * scale_x)
                    y2 = int((y + 1) * scale_y)

                    thickness = 1 if max_prob < 0.7 else 2
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), base_color, thickness)

                    label_text = f"{label_name} {max_prob:.2f}"

                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(annotated, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), base_color, -1)
                    cv2.putText(annotated, label_text, (x1 + 2, y1 - 3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    detections.append({
                        'class': label_name,
                        'confidence': float(max_prob),
                        'x': center_x,
                        'y': center_y,
                        'box': [x1, y1, x2, y2]
                    })

    total_pixels = grid_h * grid_w
    class_percentages = class_counts / total_pixels

    # Legenda
    legend_y = 30
    for i, label_name in enumerate(labels):
        if i == 0:
            continue
        base_color = COLORS_BGR.get(label_name, (255, 255, 255))
        perc = class_percentages[i] * 100
        thresh = CLASS_THRESHOLDS[label_name]
        text = f"{label_name}: {perc:.1f}% (th:{thresh:.2f})"
        cv2.putText(annotated, text, (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 2)
        legend_y += 25

    # Contador de detecções
    det_by_class = {'car': 0, 'sign': 0}
    for det in detections:
        det_by_class[det['class']] += 1

    det_text = f"Det: {len(detections)} (C:{det_by_class['car']} S:{det_by_class['sign']})"
    (text_w, text_h), _ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(annotated, det_text, (img_w - text_w - 10, img_h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Adicionar overlay de sistema
    annotated = draw_system_overlay(annotated, metrics_data)

    return class_percentages, annotated, detections

def classification_worker():
    global metrics, annotated_frame

    interpreter = load_model()
    print("✓ Thread de classificação iniciada\n")

    while True:
        if is_classifying:
            with frame_lock:
                if current_frame is not None:
                    frame_copy = current_frame
                else:
                    time.sleep(0.1)
                    continue

            try:
                # Marcar início do pipeline completo
                pipeline_start = time.time()

                # Decodificar JPEG
                nparr = np.frombuffer(frame_copy, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Copiar métricas atuais
                with inference_lock:
                    metrics_copy = metrics.copy()

                # Inferência
                inference_start = time.time()
                predictions, annotated_bgr, detections = detect_objects_fomo(
                    None, interpreter, img_bgr, metrics_copy)
                inference_time = time.time() - inference_start

                # Tempo total do pipeline (captura já foi feita antes)
                pipeline_time = time.time() - pipeline_start

                # Encode
                success, buffer = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with frame_lock:
                    annotated_frame = buffer.tobytes()

                # Atualizar métricas
                with inference_lock:
                    metrics['total_inference_time'] += inference_time
                    metrics['inference_count'] += 1
                    metrics['avg_inference_ms'] = (metrics['total_inference_time'] / metrics['inference_count']) * 1000
                    metrics['last_pipeline_time'] = pipeline_time * 1000

                    # Calcular tempo total médio do pipeline
                    # Pipeline = tempo de inferência + overhead
                    metrics['avg_total_pipeline_ms'] = metrics['last_pipeline_time']

                    # Calcular slack disponível
                    metrics['slack_ms'] = FRAME_BUDGET_MS - metrics['avg_total_pipeline_ms']

                    # FPS com inferência
                    if metrics['avg_total_pipeline_ms'] > 0:
                        metrics['fps_with_inference'] = 1000.0 / metrics['avg_total_pipeline_ms']

                    for det in detections:
                        metrics['total_detections'][det['class']] += 1

                # Preparar resultado
                max_idx = np.argmax(predictions)
                max_prob = predictions[max_idx]

                if max_idx == 0:
                    non_bg = predictions[1:]
                    if len(non_bg) > 0:
                        best_idx = np.argmax(non_bg) + 1
                        label_name = labels[best_idx]
                        if predictions[best_idx] >= CLASS_THRESHOLDS[label_name]:
                            label = label_name
                            max_prob = predictions[best_idx]
                        else:
                            label = f'Background ({len(detections)} objects)'
                    else:
                        label = 'No objects'
                else:
                    label = labels[max_idx]

                result = {
                    'label': label,
                    'probability': float(max_prob),
                    'detections_count': len(detections),
                    'all_predictions': {labels[i]: float(predictions[i]) for i in range(len(labels))}
                }

                try:
                    classification_queue.put_nowait(result)
                except:
                    pass

            except Exception as e:
                print(f"Erro na inferência: {e}")
                import traceback
                traceback.print_exc()
        else:
            with frame_lock:
                annotated_frame = None

        time.sleep(0.1)

def capture_frames():
    """Thread de captura - calcula FPS sem inferência"""
    global current_frame, metrics

    metrics['start_time'] = time.time()
    last_system_check = time.time()

    while True:
        try:
            capture_start = time.time()
            array = picam2.capture_array()  # BGR array
            capture_time = time.time() - capture_start

            encode_start = time.time()
            success, buffer = cv2.imencode('.jpg', array, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            encode_time = time.time() - encode_start

            # Se não estiver classificando, adicionar overlay nas métricas
            if not is_classifying:
                # Desenhar overlay diretamente no frame de captura
                with inference_lock:
                    metrics_copy = metrics.copy()

                frame_with_overlay = draw_system_overlay(array.copy(), metrics_copy)
                success, buffer = cv2.imencode('.jpg', frame_with_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()

            with frame_lock:
                current_frame = frame_bytes

            metrics['frame_count'] += 1
            metrics['total_capture_time'] += capture_time
            metrics['total_encode_time'] += encode_time

            current_time = time.time()

            # Atualizar métricas de sistema
            if current_time - last_system_check >= 2.0:
                metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
                metrics['cpu_temp'] = get_cpu_temperature()
                last_system_check = current_time

            # Atualizar FPS de captura
            if current_time - metrics['last_update'] >= 1.0:
                elapsed = current_time - metrics['start_time']
                metrics['fps_capture'] = metrics['frame_count'] / elapsed
                metrics['avg_capture_ms'] = (metrics['total_capture_time'] / metrics['frame_count']) * 1000
                metrics['avg_encode_ms'] = (metrics['total_encode_time'] / metrics['frame_count']) * 1000
                metrics['last_update'] = current_time

            time.sleep(0.01)

        except Exception as e:
            print(f"Erro na captura: {e}")
            time.sleep(0.1)

def generate_frames():
    while True:
        with frame_lock:
            if annotated_frame is not None:
                frame_to_send = annotated_frame
            elif current_frame is not None:
                frame_to_send = current_frame
            else:
                time.sleep(0.04)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
        time.sleep(0.04)

# Rotas Flask
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>FOMO Detection - Timing Analysis</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        .video-container img {
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .panel {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .panel h3 {
            color: #667eea;
            margin: 0 0 15px 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .slider-group {
            margin: 15px 0;
        }
        .slider-group label {
            display: block;
            margin-bottom: 5px;
            color: #666;
            font-size: 0.9em;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            outline: none;
            background: #ddd;
        }
        .slider-value {
            font-weight: bold;
            color: #667eea;
            min-width: 45px;
            text-align: right;
        }
        .btn {
            padding: 12px 20px;
            margin: 5px 0;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s;
        }
        .btn-start { background: #4CAF50; color: white; }
        .btn-start:hover { background: #45a049; }
        .btn-start:disabled { background: #ccc; cursor: not-allowed; }
        .btn-stop { background: #f44336; color: white; }
        .btn-stop:hover { background: #da190b; }
        .btn-stop:disabled { background: #ccc; cursor: not-allowed; }
        .result {
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .detection-count {
            font-size: 1.1em;
            color: #4CAF50;
            margin: 5px 0;
        }
        .predictions {
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }
        .prediction-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .metric {
            font-size: 1.05em;
            margin: 10px 0;
            color: #555;
        }
        .metric strong { color: #667eea; }
        .metric-good { color: #4CAF50 !important; }
        .metric-warn { color: #FF9800 !important; }
        .metric-bad { color: #f44336 !important; }
        .info-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            margin: 10px 0;
            font-size: 0.85em;
            border-radius: 4px;
        }
        @media (max-width: 1024px) {
            .content { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 FOMO Detection - Análise de Timing</h1>
        <div class="subtitle">Métricas detalhadas de FPS e Pipeline</div>

        <div class="content">
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Camera Stream">
            </div>

            <div>
                <div class="panel">
                    <h3>🎚️ Thresholds</h3>

                    <div class="slider-group">
                        <label>🟢 Car:</label>
                        <div class="slider-container">
                            <input type="range" min="0" max="100" value="25" class="slider" id="carSlider">
                            <span class="slider-value" id="carValue">0.25</span>
                        </div>
                    </div>

                    <div class="slider-group">
                        <label>🔴 Sign:</label>
                        <div class="slider-container">
                            <input type="range" min="0" max="100" value="35" class="slider" id="signSlider">
                            <span class="slider-value" id="signValue">0.35</span>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h3>🎯 Controle</h3>
                    <button class="btn btn-start" id="startBtn" onclick="startClassification()">▶️ Iniciar</button>
                    <button class="btn btn-stop" id="stopBtn" onclick="stopClassification()" disabled>⏹️ Parar</button>
                </div>

                <div class="panel">
                    <h3>📊 Resultado</h3>
                    <div class="result" id="result">Aguardando...</div>
                    <div class="detection-count" id="detCount"></div>
                    <div class="predictions" id="allPreds"></div>
                </div>

                <div class="panel">
                    <h3>⚡ Métricas Detalhadas</h3>
                    <div class="info-box">
                        Budget: 41.67ms para 24 FPS
                    </div>
                    <div class="metric">FPS Captura: <strong id="fpsCap">--</strong></div>
                    <div class="metric">FPS + Inferência: <strong id="fpsInf">--</strong></div>
                    <div class="metric">Pipeline Total: <strong id="pipeline" class="metric-good">--</strong> ms</div>
                    <div class="metric">Slack Disponível: <strong id="slack" class="metric-good">--</strong> ms</div>
                    <div class="metric">CPU: <strong id="cpu">--</strong>%</div>
                    <div class="metric">Temp: <strong id="temp">--</strong>°C</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let classificationActive = false;

        document.getElementById('carSlider').oninput = function() {
            const value = (this.value / 100).toFixed(2);
            document.getElementById('carValue').textContent = value;
            updateThresholds();
        }

        document.getElementById('signSlider').oninput = function() {
            const value = (this.value / 100).toFixed(2);
            document.getElementById('signValue').textContent = value;
            updateThresholds();
        }

        function updateThresholds() {
            const thresholds = {
                'car': parseFloat(document.getElementById('carValue').textContent),
                'sign': parseFloat(document.getElementById('signValue').textContent)
            };

            fetch('/update_thresholds', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({thresholds: thresholds})
            });
        }

        function startClassification() {
            fetch('/start_classification', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    classificationActive = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    updateClassification();
                });
        }

        function stopClassification() {
            fetch('/stop_classification', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    classificationActive = false;
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    document.getElementById('result').textContent = 'Parado';
                    document.getElementById('detCount').textContent = '';
                    document.getElementById('allPreds').innerHTML = '';
                });
        }

        function updateClassification() {
            if (!classificationActive) return;
            fetch('/get_classification')
                .then(r => r.json())
                .then(data => {
                    if (data.label) {
                        document.getElementById('result').textContent = data.label;
                        document.getElementById('detCount').textContent = 
                            '🎯 ' + data.detections_count + ' objeto(s)';

                        let html = '<strong>Cobertura:</strong>';
                        for (const [label, prob] of Object.entries(data.all_predictions)) {
                            html += '<div class="prediction-item">';
                            html += '<span>' + label + '</span>';
                            html += '<span>' + (prob * 100).toFixed(1) + '%</span>';
                            html += '</div>';
                        }
                        document.getElementById('allPreds').innerHTML = html;
                    }
                });
            setTimeout(updateClassification, 500);
        }

        function updateMetrics() {
            fetch('/metrics')
                .then(r => r.json())
                .then(data => {
                    // FPS
                    document.getElementById('fpsCap').textContent = data.fps_capture.toFixed(2);
                    document.getElementById('fpsInf').textContent = 
                        data.fps_with_inference > 0 ? data.fps_with_inference.toFixed(2) : 'N/A';

                    // Pipeline
                    const pipelineElem = document.getElementById('pipeline');
                    if (data.avg_total_pipeline_ms > 0) {
                        pipelineElem.textContent = data.avg_total_pipeline_ms.toFixed(2);
                        pipelineElem.className = data.avg_total_pipeline_ms < 35 ? 'metric-good' : 
                                                 data.avg_total_pipeline_ms < 41.67 ? 'metric-warn' : 'metric-bad';
                    } else {
                        pipelineElem.textContent = 'N/A';
                        pipelineElem.className = '';
                    }

                    // Slack
                    const slackElem = document.getElementById('slack');
                    if (data.slack_ms !== 0 || classificationActive) {
                        slackElem.textContent = data.slack_ms.toFixed(2);
                        slackElem.className = data.slack_ms > 5 ? 'metric-good' : 
                                             data.slack_ms > 0 ? 'metric-warn' : 'metric-bad';
                    } else {
                        slackElem.textContent = 'N/A';
                        slackElem.className = '';
                    }

                    // CPU e Temp
                    document.getElementById('cpu').textContent = data.cpu_percent.toFixed(1);
                    document.getElementById('temp').textContent = data.cpu_temp.toFixed(1);
                });
        }

        setInterval(updateMetrics, 500);
        updateMetrics();
    </script>
</body>
</html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def get_metrics():
    with inference_lock:
        return jsonify({
            'frame_count': metrics['frame_count'],
            'fps_capture': metrics['fps_capture'],
            'fps_with_inference': metrics['fps_with_inference'],
            'avg_capture_ms': metrics['avg_capture_ms'],
            'avg_encode_ms': metrics['avg_encode_ms'],
            'avg_inference_ms': metrics['avg_inference_ms'],
            'avg_total_pipeline_ms': metrics['avg_total_pipeline_ms'],
            'slack_ms': metrics['slack_ms'],
            'cpu_percent': metrics['cpu_percent'],
            'cpu_temp': metrics['cpu_temp'],
            'total_detections': metrics['total_detections']
        })

@app.route('/start_classification', methods=['POST'])
def start_classification():
    global is_classifying
    is_classifying = True
    metrics['total_detections'] = {'car': 0, 'sign': 0}
    return jsonify({'status': 'started'})

@app.route('/stop_classification', methods=['POST'])
def stop_classification():
    global is_classifying
    is_classifying = False
    return jsonify({'status': 'stopped'})

@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    global CLASS_THRESHOLDS
    data = request.get_json()
    thresholds = data.get('thresholds', {})

    for cls, thresh in thresholds.items():
        if cls in CLASS_THRESHOLDS:
            CLASS_THRESHOLDS[cls] = thresh

    return jsonify({'status': 'updated', 'thresholds': CLASS_THRESHOLDS})

@app.route('/get_classification')
def get_classification():
    try:
        result = classification_queue.get_nowait()
        return jsonify(result)
    except:
        return jsonify({'label': None})

if __name__ == '__main__':
    print("="*70)
    print("FOMO DETECTION - ANÁLISE DETALHADA DE TIMING")
    print("="*70)

    if not initialize_camera():
        print("❌ Erro ao inicializar câmera")
        exit(1)

    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    classification_thread = threading.Thread(target=classification_worker, daemon=True)
    classification_thread.start()

    print(f"\n📋 Configuração:")
    print(f"  • Target FPS: {TARGET_FPS}")
    print(f"  • Frame Budget: {FRAME_BUDGET_MS:.2f}ms")
    print(f"  • Resolução: {RESOLUTION}")

    print(f"\n📊 Métricas Monitoradas:")
    print(f"  • FPS Captura (sem inferência)")
    print(f"  • FPS com Inferência (pipeline completo)")
    print(f"  • Tempo Total Pipeline")
    print(f"  • Slack Disponível (budget - pipeline)")
    print(f"  • CPU e Temperatura")

    print(f"\n🌐 Servidor iniciado!")
    print(f"  Acesse: http://localhost:5000")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
