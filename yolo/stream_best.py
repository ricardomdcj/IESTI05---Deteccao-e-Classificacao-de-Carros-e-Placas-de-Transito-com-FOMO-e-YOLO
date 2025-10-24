from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import io
import threading
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ncnn
from collections import defaultdict
from picamera2 import Picamera2
import atexit
import psutil

app = Flask(__name__)
camera = None

camera = None
picam2 = None
last_raw_pil = None
frame_jpeg = None
frame_lock = threading.Lock()
is_detecting = False
confidence_threshold = 0.5
latest_detections = []
detections_lock = threading.Lock()

# Variáveis para calcular FPS
fps_counter = 0
fps_start_time = time.time()
current_fps = 0.0

ncnn_net = None
# Corrigir caminhos: .param deve apontar para o arquivo param e .bin para o bin
MODEL_PARAM = './models/best_ncnn_model/model.ncnn.param'
MODEL_BIN   = './models/best_ncnn_model/model.ncnn.bin'
INPUT_SIZE = 640

# Ajuste labels para o seu conjunto de classes real
# MODIFICADO: labels agora mapeiam class_0 -> "car" e class_1 -> "sign"
labels = {0: 'car', 1: 'sign'}
for i in range(2, 80):
    labels[i] = f'class_{i}'

# IoU de deduplicação
counts_accum_total = 0
counts_accum_per_class = defaultdict(int)
counted_boxes = []
counted_classes = []

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB-xA+1) * max(0, yB-yA+1)
    boxAArea = (boxA[2]-boxA[0]+1) * (boxA[3]-boxA[1]+1)
    boxBArea = (boxB[2]-boxB[0]+1) * (boxB[3]-boxB[1]+1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def letterbox_pil(img_pil, new_shape=(640, 640), color=(114, 114, 114)):
    w0, h0 = img_pil.size
    r = min(new_shape[0] / w0, new_shape[1] / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_shape[0] - new_unpad[0]) // 2
    dh = (new_shape[1] - new_unpad[1]) // 2
    img_resized = img_pil.resize(new_unpad, Image.BILINEAR)
    canvas = Image.new("RGB", new_shape, color)
    canvas.paste(img_resized, (dw, dh))
    return canvas, r, dw, dh

def xywh2xyxy(xywh):
    x, y, w, h = xywh
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

def nms_boxes(boxes, scores, iou_thres=0.45):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        i_box = boxes[i]
        rest = boxes[idxs[1:]]
        xx1 = np.maximum(i_box[0], rest[:, 0])
        yy1 = np.maximum(i_box[1], rest[:, 1])
        xx2 = np.minimum(i_box[2], rest[:, 2])
        yy2 = np.minimum(i_box[3], rest[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (i_box[2] - i_box[0]) * (i_box[3] - i_box[1])
        area_r = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = idxs[1:][iou <= iou_thres]
    return keep

def initialize_camera():
    global camera, picam2
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)
    camera = True


def stop_camera():
    global picam2
    try:
        if picam2 is not None:
            picam2.stop()
            picam2.close()
    except Exception:
        pass

atexit.register(stop_camera)

def get_cpu_temp():
    """Obtém a temperatura da CPU em Celsius"""
    try:
        # Para Raspberry Pi
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
        return temp
    except:
        return 0.0

def get_frame():
    global frame_jpeg, last_raw_pil, fps_counter, fps_start_time, current_fps
    while True:
        frame_rgb = picam2.capture_array()
        raw_pil = Image.fromarray(frame_rgb)

        img_draw = draw_detections(raw_pil.copy())

        buf = io.BytesIO()
        img_draw.save(buf, format='JPEG', quality=80)

        with frame_lock:
            last_raw_pil = raw_pil
            frame_jpeg = buf.getvalue()

        # Calcular FPS
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()

        time.sleep(0.03)


def generate_frames():
    while True:
        with frame_lock:
            if frame_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')
        time.sleep(0.01)

def load_ncnn_model():
    global ncnn_net
    if ncnn_net is None:
        ncnn_net = ncnn.Net()
        ncnn_net.load_param(MODEL_PARAM)
        ncnn_net.load_model(MODEL_BIN)
    return ncnn_net

# Tenta múltiplos nomes conhecidos para input/output dos exports NCNN
def try_input(ex, mat):
    for name in ["in0", "images", "data", "input.1"]:
        try:
            if ex.input(name, mat) == 0:
                print(f"[NCNN] Using input blob: {name}")
                return name
        except Exception:
            pass
    raise RuntimeError("Nenhum blob de entrada conhecido encontrado (tente verificar o .param).")

def extract_any(ex):
    for name in ["out0", "output0", "output", "prob"]:
        try:
            ret, mat = ex.extract(name)
            if ret == 0 and getattr(mat, 'elemsize', 0) in (1, 2, 4):
                print(f"[NCNN] Using output blob: {name} (elemsize={mat.elemsize})")
                return mat, name
        except Exception:
            pass
    raise RuntimeError("Nenhum blob de saída conhecido encontrado ou elemsize inválido (verifique nomes no .param).")

def detect_objects_ncnn(img_pil, net, conf_thres=0.5, iou_thres=0.45):
    lb_img, r, dw, dh = letterbox_pil(img_pil, (INPUT_SIZE, INPUT_SIZE), color=(114, 114, 114))
    np_lb = np.array(lb_img, dtype=np.uint8)
    H, W, _ = np_lb.shape
    mat_in = ncnn.Mat.from_pixels(np_lb.tobytes(), ncnn.Mat.PixelType.PIXEL_RGB, W, H)
    mean = (0.0, 0.0, 0.0)
    norm = (1/255.0, 1/255.0, 1/255.0)
    mat_in.substract_mean_normalize(mean, norm)

    ex = net.create_extractor()
    try_input(ex, mat_in)
    out_mat, out_name = extract_any(ex)

    if getattr(out_mat, 'elemsize', 0) not in (1, 2, 4):
        raise RuntimeError(f"Output elemsize inválido: {getattr(out_mat, 'elemsize', None)} do blob {out_name}")

    out_np = np.array(out_mat, dtype=np.float32)
    if out_np.ndim == 1:
        out_np = out_np.reshape(1, -1)
    elif out_np.ndim == 2 and out_np.shape[0] < out_np.shape[1]:
        out_np = out_np.T
    elif out_np.ndim >= 3:
        out_np = out_np.reshape(out_np.shape[0], -1)

    C = out_np.shape[1]
    if C < 6:
        return []

    # Detect head: com ou sem objectness, conforme export
    L = max(labels.keys()) + 1 if isinstance(labels, dict) else 80
    has_obj = (C == (4 + 1 + L))
    if has_obj:
        num_classes = L
        cls_offset = 5
    else:
        num_classes = C - 4
        cls_offset = 4

    boxes, scores, classes = [], [], []
    for row in out_np:
        if has_obj:
            obj = float(row[4])
            cls_scores = row[cls_offset:cls_offset + num_classes]
            cid = int(np.argmax(cls_scores))
            score = obj * float(cls_scores[cid])
        else:
            cls_scores = row[cls_offset:cls_offset + num_classes]
            cid = int(np.argmax(cls_scores))
            score = float(cls_scores[cid])

        if score < conf_thres:
            continue

        x1, y1, x2, y2 = xywh2xyxy(row[:4])
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(cid)

    if not boxes:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    classes = np.array(classes, dtype=np.int32)

    keep_all = []
    for cid in np.unique(classes):
        idxs = np.where(classes == cid)[0]
        kept = nms_boxes(boxes[idxs], scores[idxs], iou_thres=iou_thres)
        keep_all.extend(idxs[kept])

    W0, H0 = img_pil.size
    detections = []
    for i in keep_all:
        x1, y1, x2, y2 = boxes[i]
        # desfaz letterbox
        x1 -= dw; y1 -= dh; x2 -= dw; y2 -= dh
        x1 /= r; y1 /= r; x2 /= r; y2 /= r
        # clamp
        x1 = max(0, min(W0 - 1, x1)); y1 = max(0, min(H0 - 1, y1))
        x2 = max(0, min(W0 - 1, x2)); y2 = max(0, min(H0 - 1, y2))
        detections.append({
            'class': labels.get(int(classes[i]), f"Class {int(classes[i])}"),
            'class_id': int(classes[i]),
            'score': float(scores[i]),
            'box': [int(x1), int(y1), int(x2), int(y2)]
        })
    return detections

def detection_worker():
    global is_detecting, latest_detections, counts_accum_total, counts_accum_per_class, counted_boxes, counted_classes
    net = load_ncnn_model()
    IOU_THRESHOLD = 0.3
    while True:
        if is_detecting:
            try:
                with frame_lock:
                    raw = last_raw_pil.copy() if last_raw_pil is not None else None
                if raw is not None:
                    detections = detect_objects_ncnn(raw, net, conf_thres=confidence_threshold, iou_thres=0.45)
                    with detections_lock:
                        latest_detections = detections
                    for det in detections:
                        box = det['box']
                        cls = det['class']
                        already_counted = False
                        for i, cbox in enumerate(counted_boxes):
                            if counted_classes[i] == cls and iou(box, cbox) > IOU_THRESHOLD:
                                already_counted = True
                                break
                        if not already_counted:
                            counts_accum_total += 1
                            counts_accum_per_class[cls] += 1
                            counted_boxes.append(box)
                            counted_classes.append(cls)
            except Exception as e:
                print(f"Erro no worker de detecção: {e}")
        time.sleep(0.01)

def draw_detections(img):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # ADICIONADO: Desenhar métricas no canto superior esquerdo
    cpu_percent = psutil.cpu_percent(interval=0)
    cpu_temp = get_cpu_temp()

    with detections_lock:
        item_count = len(latest_detections)

        # Posição inicial para as métricas
        metrics_x = 10
        metrics_y = 10
        line_height = 18

        # Criar textos das métricas
        metrics = [
            f"FPS: {current_fps:.1f}",
            f"CPU: {cpu_percent:.1f}%",
            f"Temp: {cpu_temp:.1f}°C",
            f"Items: {item_count}"
        ]

        # Desenhar cada métrica com fundo semi-transparente
        for i, metric_text in enumerate(metrics):
            y_pos = metrics_y + (i * line_height)

            # Calcular tamanho do texto para criar fundo
            bbox = draw.textbbox((metrics_x, y_pos), metric_text, font=font_small)

            # Desenhar retângulo de fundo (preto semi-transparente)
            # Como PIL não suporta alpha diretamente, usamos preto sólido
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="black")

            # Desenhar texto em branco
            draw.text((metrics_x, y_pos), metric_text, font=font_small, fill="white")

        # Desenhar detecções (código original)
        for det in latest_detections:
            l, t, r, b = det['box']
            draw.rectangle([l, t, r, b], outline="red", width=2)
            label = f"{det['class']}: {det['score']:.2f}"
            draw.text((l, max(0, t - 15)), label, font=font, fill="red")

    return img

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection and Counting</title>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script>
                function startDetection() {
                    $.post('/start');
                    $('#startBtn').prop('disabled', true);
                    $('#stopBtn').prop('disabled', false);
                }
                function stopDetection() {
                    $.post('/stop');
                    $('#startBtn').prop('disabled', false);
                    $('#stopBtn').prop('disabled', true);
                }
                function resetCounts() {
                    $.post('/reset_counts', function(response) {
                        alert(response);
                    });
                }
                function updateInfo() {
                    $.get('/get_detections', function(data) {
                        $('#detections').empty();
                        const stable = data.sort((a, b) => b.score - a.score);
                        stable.forEach(d => {
                            $('#detections').append(`<p>${d.class}: ${d.score.toFixed(2)}</p>`);
                        });
                    });
                    $.get('/counts_json', function(stats) {
                        $('#counts').text('Total acumulado: ' + stats.total);
                        let perClass = '';
                        Object.entries(stats.per_class).forEach(([k,v])=>{
                            perClass += `<span>${k}: ${v} </span>`;
                        });
                        $('#counts_per_class').html(perClass);
                    });
                }
                $(document).ready(function() {
                    setInterval(updateInfo, 700);
                });
            </script>
        </head>
        <body>
            <h1>Object Detection and Counting</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480" />
            <br>
            <button id="startBtn" onclick="startDetection()">Start Detection</button>
            <button id="stopBtn" onclick="stopDetection()" disabled>Stop Detection</button>
            <button onclick="resetCounts()">Reset Count</button>
            <br>
            <label for="confidence">Confidence Threshold:</label>
            <input type="number" id="confidence" name="confidence" min="0" max="1" step="0.05" value="0.5" onchange="$.post('/update_confidence', {confidence: this.value});">
            <h3>Contagens Acumuladas</h3>
            <div id="counts">Total acumulado: 0</div>
            <div id="counts_per_class"></div>
            <h3>Detecções Atuais</h3>
            <div id="detections">Aguardando detecções...</div>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts_json')
def counts_json():
    return jsonify({
        "total": counts_accum_total,
        "per_class": dict(counts_accum_per_class)
    })

@app.route('/reset_counts', methods=['POST'])
def reset_counts():
    global counts_accum_total, counts_accum_per_class, counted_boxes, counted_classes
    counts_accum_total = 0
    counts_accum_per_class.clear()
    counted_boxes.clear()
    counted_classes.clear()
    return "Contagens resetadas", 200

@app.route('/start', methods=['POST'])
def start_detection():
    global is_detecting
    is_detecting = True
    return '', 204

@app.route('/stop', methods=['POST'])
def stop_detection():
    global is_detecting
    is_detecting = False
    return '', 204

@app.route('/update_confidence', methods=['POST'])
def update_confidence():
    global confidence_threshold
    confidence_threshold = float(request.form['confidence'])
    return '', 204

@app.route('/get_detections')
def get_detections():
    global latest_detections
    if not is_detecting:
        return jsonify([])
    with detections_lock:
        return jsonify(latest_detections)


if __name__ == '__main__':
    try:
        # 1. Inicializa a câmera
        print("Inicializando a câmera...")
        initialize_camera()

        # 2. Inicia a thread que captura frames da câmera
        print("Iniciando thread de captura de frames...")
        frame_thread = threading.Thread(target=get_frame)
        frame_thread.daemon = True
        frame_thread.start()

        # 3. Inicia a thread que faz a detecção de objetos
        print("Iniciando thread de detecção NCNN...")
        detection_thread = threading.Thread(target=detection_worker)
        detection_thread.daemon = True
        detection_thread.start()

        # 4. Inicia o servidor Flask
        print("Iniciando servidor Flask em http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)

    except KeyboardInterrupt:
        print("Encerrando...")
    finally:
        # Garante que a câmera seja desligada ao sair
        stop_camera()
