import time
from picamera2 import Picamera2
from datetime import datetime
import os
import numpy as np
import signal
import sys
import cv2
import ncnn
import psutil
import math
from pytz import timezone
import threading                           
from queue import Queue, Empty             

# --- Configurações Gerais ---
RESOLUTION = (640, 480)
TZ_BR = timezone('America/Sao_Paulo')
folder_time = datetime.now(TZ_BR).strftime("%Y%m%d_%H%M%S")
SAVE_DIR = folder_time
os.makedirs(SAVE_DIR, exist_ok=True)
JPEG_QUALITY = 85

# --- Configurações do Modelo NCNN ---
MODEL_PARAM = './models/best_ncnn_model/model.ncnn.param'
MODEL_BIN   = './models/best_ncnn_model/model.ncnn.bin'
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

labels = {
    0: 'car',
    1: 'sign'
}

# --- Configurações do Rastreador/Contador ---
active_tracks = []      
next_object_id = 0      
total_car_count = 0     
total_sign_count = 0    
MAX_ASSOCIATION_DIST = 200
MAX_FRAMES_TO_LIVE = 3

# --- Fila e Sinal de Parada para a Thread de Salvamento ---
save_queue = Queue(maxsize=30) 
stop_event = threading.Event()

# --- Variáveis Globais ---
picam2 = None
ncnn_net = None
frame_count = 0
loop_start_time = time.time()

# --- Listas para armazenar métricas para o relatório  ---
all_fps_readings = []
all_cpu_readings = []
all_temp_readings = []
all_mem_readings = []
all_swap_readings = []
all_loop_ms_readings = []
global_start_time = None

# --- Thread Worker para Salvar Imagens ---

def save_worker():
    """Thread 'worker' que salva imagens da fila em segundo plano."""
    print("[Save Thread] Iniciada.")
    while not stop_event.is_set():
        try:
            frame_data = save_queue.get(timeout=1)
            filename = frame_data['filename']
            image_bgr = frame_data['image']
            
            cv2.imwrite(filename, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            save_queue.task_done()
            
        except Empty:
            continue
        except Exception as e:
            print(f"[Save Thread] Erro ao salvar imagem {filename}: {e}")
            
    print("[Save Thread] Finalizada.")


# --- Funções de Utilitários ---
def get_center(box):
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    return (cx, cy)

def calc_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", 'r') as f:
            temp_milli = int(f.read().strip())
            return temp_milli / 1000.0
    except Exception as e:
        return -1.0

def get_cpu_usage():
    return psutil.cpu_percent(interval=None)
    
def get_memory_usage():
    return psutil.virtual_memory().percent

def get_swap_usage():
    return psutil.swap_memory().percent

# --- Funções de Finalização ---

def signal_handler(sig, frame):
    print("\n\nCaptura interrompida pelo usuário (Ctrl+C)")
    sys.exit(0)

def finalize_capture():
    global picam2, stop_event
    print("Encerrando a câmera...")
    if picam2 is not None:
        try: picam2.stop(); picam2.close()
        except: pass
        
    print("Sinalizando para a thread de salvamento parar...")
    stop_event.set()
    
    print(f"Aguardando {save_queue.qsize()} imagens na fila de salvamento...")
    save_queue.join() 
    
    print("Gerando relatório final...")
    
    if frame_count > 0 and global_start_time is not None:
        end_time = datetime.now(TZ_BR)
        total_duration_sec = (end_time - global_start_time).total_seconds()
        
        avg_fps = np.mean(all_fps_readings) if all_fps_readings else 0
        max_fps = np.max(all_fps_readings) if all_fps_readings else 0
        min_fps = np.min(all_fps_readings) if all_fps_readings else 0
        avg_cpu = np.mean(all_cpu_readings) if all_cpu_readings else 0
        max_cpu = np.max(all_cpu_readings) if all_cpu_readings else 0
        min_cpu = np.min(all_cpu_readings) if all_cpu_readings else 0
        avg_temp = np.mean(all_temp_readings) if all_temp_readings else 0
        max_temp = np.max(all_temp_readings) if all_temp_readings else 0
        min_temp = np.min(all_temp_readings) if all_temp_readings else 0
        avg_mem = np.mean(all_mem_readings) if all_mem_readings else 0
        max_mem = np.max(all_mem_readings) if all_mem_readings else 0
        min_mem = np.min(all_mem_readings) if all_mem_readings else 0
        avg_swap = np.mean(all_swap_readings) if all_swap_readings else 0
        max_swap = np.max(all_swap_readings) if all_swap_readings else 0
        min_swap = np.min(all_swap_readings) if all_swap_readings else 0
        avg_loop_ms = np.mean(all_loop_ms_readings) if all_loop_ms_readings else 0
        max_loop_ms = np.max(all_loop_ms_readings) if all_loop_ms_readings else 0
        min_loop_ms = np.min(all_loop_ms_readings) if all_loop_ms_readings else 0
        
        report_content = f"""
==================================================
RELATÓRIO DA SESSÃO DE CAPTURA E CLASSIFICAÇÃO
==================================================

Diretório de Salvamento: {SAVE_DIR}
Data de Início: {global_start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
Data de Término: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
Duração Total: {total_duration_sec:.2f} segundos
Total de Frames Processados: {frame_count}

--- ESTATÍSTICAS DE CONTAGEM ---
Total de Carros Contados: {total_car_count}
Total de Placas Contadas: {total_sign_count}

--- ESTATÍSTICAS DE DESEMPENHO (Média / Mín / Máx) ---

Tempo de Processamento (Loop):
  - Média: {avg_loop_ms:.0f} ms
  - Mínimo: {min_loop_ms:.0f} ms
  - Máximo: {max_loop_ms:.0f} ms

FPS (Frames Por Segundo):
  - Média: {avg_fps:.2f} FPS
  - Mínimo: {min_fps:.2f} FPS
  - Máximo: {max_fps:.2f} FPS

Uso da CPU:
  - Média: {avg_cpu:.1f} %
  - Mínimo: {min_cpu:.1f} %
  - Máximo: {max_cpu:.1f} %

Uso de Memória (RAM):
  - Média: {avg_mem:.1f} %
  - Mínimo: {min_mem:.1f} %
  - Máximo: {max_mem:.1f} %

Uso de Swap:
  - Média: {avg_swap:.1f} %
  - Mínimo: {min_swap:.1f} %
  - Máximo: {max_swap:.1f} %

Temperatura da CPU:
  - Média: {avg_temp:.1f} °C
  - Mínima: {min_temp:.1f} °C
  - Máxima: {max_temp:.1f} °C

--- CONFIGURAÇÕES ---
Resolução: {RESOLUTION[0]}x{RESOLUTION[1]}
Modelo NCNN: {MODEL_PARAM}
NCNN Threads: 4
Limite de Confiança: {CONFIDENCE_THRESHOLD}
Tolerância de Distância (Track): {MAX_ASSOCIATION_DIST} px
Tempo de Vida (Track): {MAX_FRAMES_TO_LIVE} frames

==================================================
"""
        
        report_path = os.path.join(SAVE_DIR, "session_report.txt")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Relatório salvo com sucesso em: {report_path}")
        except Exception as e:
            print(f"Erro ao salvar relatório: {e}")
            
    else:
        print("Nenhum frame foi processado. Nenhum relatório gerado.")

    print("Captura finalizada.")

signal.signal(signal.SIGINT, signal_handler)


# --- Funções de Inferência NCNN ---

def letterbox_cv2(img_rgb, new_shape=(640, 640), color=(114, 114, 114)):
    """Redimensiona e aplica padding (letterbox) usando OpenCV."""
    h0, w0, _ = img_rgb.shape
    r = min(new_shape[0] / w0, new_shape[1] / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    
    img_resized = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    dw = (new_shape[0] - new_unpad[0]) // 2
    dh = (new_shape[1] - new_unpad[1]) // 2
    
    top, bottom = dh, new_shape[1] - new_unpad[1] - dh
    left, right = dw, new_shape[0] - new_unpad[0] - dw
    
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, dw, dh

def xywh2xyxy(xywh):
    x, y, w, h = xywh
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

def nms_boxes(boxes, scores, iou_thres=0.45):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1: break
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

def load_ncnn_model():
    net = ncnn.Net()
    
    net.opt.num_threads = 4 
    # net.opt.use_vulkan_compute = False 
    
    net.load_param(MODEL_PARAM)
    net.load_model(MODEL_BIN)
    print(f"[NCNN] Modelo carregado com net.opt.num_threads = 4.")
    return net

def try_input(ex, mat):
    for name in ["in0", "images", "data", "input.1"]:
        try:
            if ex.input(name, mat) == 0:
                print(f"[NCNN] Usando blob de entrada: {name}")
                return name
        except Exception: pass
    raise RuntimeError("Nenhum blob de entrada NCNN conhecido encontrado.")

def extract_any(ex):
    for name in ["out0", "output0", "output", "prob"]:
        try:
            ret, mat = ex.extract(name)
            if ret == 0 and getattr(mat, 'elemsize', 0) in (1, 2, 4):
                print(f"[NCNN] Usando blob de saída: {name}")
                return mat, name
        except Exception: pass
    raise RuntimeError("Nenhum blob de saída NCNN conhecido encontrado.")

def detect_objects_ncnn(img_array_rgb, net, conf_thres=0.5, iou_thres=0.45):
    """Recebe array numpy RGB e usa letterbox_cv2."""
    
    lb_img_np, r, dw, dh = letterbox_cv2(img_array_rgb, (INPUT_SIZE, INPUT_SIZE), color=(114, 114, 114))
    
    H, W, _ = lb_img_np.shape
    mat_in = ncnn.Mat.from_pixels(lb_img_np.tobytes(), ncnn.Mat.PixelType.PIXEL_RGB, W, H)
    
    mean = (0.0, 0.0, 0.0)
    norm = (1/255.0, 1/255.0, 1/255.0)
    mat_in.substract_mean_normalize(mean, norm)
    
    ex = net.create_extractor()
    
    try:
        if not hasattr(detect_objects_ncnn, 'input_name'):
            detect_objects_ncnn.input_name = try_input(ex, mat_in)
        else:
            ex.input(detect_objects_ncnn.input_name, mat_in)
        if not hasattr(detect_objects_ncnn, 'output_name'):
             _, detect_objects_ncnn.output_name = extract_any(ex)
    except RuntimeError as e:
        print(f"Erro ao configurar extractor: {e}")
        return []
    
    ret, out_mat = ex.extract(detect_objects_ncnn.output_name)
    if ret != 0: return []
    if getattr(out_mat, 'elemsize', 0) not in (1, 2, 4): return []
    
    out_np = np.array(out_mat, dtype=np.float32)
    
    if out_np.ndim == 1: out_np = out_np.reshape(1, -1)
    elif out_np.ndim == 2 and out_np.shape[0] < out_np.shape[1]: out_np = out_np.T
    elif out_np.ndim >= 3: out_np = out_np.reshape(out_np.shape[0], -1)
    C = out_np.shape[1]
    if C < 6: return []
    L = max(labels.keys()) + 1 if isinstance(labels, dict) else 80
    has_obj = (C == (4 + 1 + L))
    if has_obj: num_classes = L; cls_offset = 5
    else: num_classes = C - 4; cls_offset = 4
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
        if score < conf_thres: continue
        x1, y1, x2, y2 = xywh2xyxy(row[:4])
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(cid)
    if not boxes: return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    classes = np.array(classes, dtype=np.int32)
    keep_all = []
    for cid in np.unique(classes):
        idxs = np.where(classes == cid)[0]
        kept = nms_boxes(boxes[idxs], scores[idxs], iou_thres=iou_thres)
        keep_all.extend(idxs[kept])
        
    H0, W0, _ = img_array_rgb.shape
    
    detections = []
    for i in keep_all:
        x1, y1, x2, y2 = boxes[i]
        x1 -= dw; y1 -= dh; x2 -= dw; y2 -= dh
        x1 /= r; y1 /= r; x2 /= r; y2 /= r
        x1 = max(0, min(W0 - 1, x1)); y1 = max(0, min(H0 - 1, y1))
        x2 = max(0, min(W0 - 1, x2)); y2 = max(0, min(H0 - 1, y2))
        detections.append({
            'class': labels.get(int(classes[i]), f"Class {int(classes[i])}"),
            'class_id': int(classes[i]),
            'score': float(scores[i]),
            'box': [int(x1), int(y1), int(x2), int(y2)]
        })
    return detections

# --- Funções de Desenho ---

def draw_detections_on_image_cv2(img_array, detections):
    """Desenha detecções (bounding boxes) na imagem usando OpenCV."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    for det in detections:
        l, t, r, b = det['box'] 
        class_name = det['class']
        label = f"{class_name}: {det['score']:.2f}"
        
        if class_name.lower() == 'sign':
            color = (0, 255, 0) # Verde (BGR)
        else:
            color = (0, 0, 255) # Vermelho (BGR)
            
        cv2.rectangle(img_array, (l, t), (r, b), color, 2) 
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        text_t_baseline = max(0, t - 5) 
        text_b_top = text_t_baseline - text_h - 4 
        
        cv2.rectangle(img_array, (l - 1, text_b_top), (l + text_w + 2, text_t_baseline + 4), color, -1)
        cv2.putText(img_array, label, (l + 1, text_t_baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return img_array 

def draw_system_stats_cv2(img_array, fps, cpu, temp, mem, swap, loop_ms, car_count, sign_count, now_timestamp):
    """Desenha estatísticas de sistema e contagem na imagem usando OpenCV."""
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 16 
    
    info_text = [
        f"FPS: {fps:.1f}",
        f"Loop: {loop_ms:.0f} ms",
        f"CPU: {cpu:.1f}%",
        f"RAM: {mem:.1f}%",
        f"Temp: {temp:.1f}C",
        f"Cars: {car_count}",
        f"Signs: {sign_count}"
    ]

    box_h = (len(info_text) * line_height) + 10
    cv2.rectangle(img_array, (0, 0), (140, box_h), (0,0,0), -1) 
    
    for i, line in enumerate(info_text):
        y_pos = (i * line_height) + 15
        cv2.putText(img_array, line, (5, y_pos), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    time_str = now_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    (time_w, time_h), _ = cv2.getTextSize(time_str, font, font_scale, font_thickness)
    
    x_pos = RESOLUTION[0] - time_w - 5 
    y_pos_baseline = 15 
    
    cv2.rectangle(img_array, (x_pos - 2, y_pos_baseline - time_h - 2), (x_pos + time_w + 2, y_pos_baseline + 5), (0,0,0), -1)
    cv2.putText(img_array, time_str, (x_pos, y_pos_baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img_array

# --- Loop Principal ---

def main():
    global picam2, ncnn_net, frame_count, loop_start_time
    global active_tracks, next_object_id, total_car_count, total_sign_count
    global global_start_time, all_fps_readings, all_cpu_readings, all_temp_readings
    global all_mem_readings, all_swap_readings, all_loop_ms_readings
    
    print("Carregando modelo NCNN...")
    ncnn_net = load_ncnn_model()

    # Inicializar câmera
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": RESOLUTION, "format": "BGR888"}, #Por alguma força misteriosa, precisa ser BGR888
        buffer_count=3
    )
    picam2.configure(config)
    picam2.start()
    print("Câmera inicializada.")
    time.sleep(2.0)
    
    # Inicia a thread de salvamento
    print("Iniciando thread de salvamento em segundo plano...")
    saver_thread = threading.Thread(target=save_worker, daemon=True)
    saver_thread.start()

    print(f"\nPasta de destino: {SAVE_DIR}/")
    print("Iniciando captura e classificação... (Pressione Ctrl+C para parar)\n")
    print("-" * 80)
    
    psutil.cpu_percent(interval=None) 
    time.sleep(0.5) 
    
    global_start_time = datetime.now(TZ_BR)
    loop_start_time = time.time()
    
    # Variáveis para "throttling" das estatísticas
    cached_cpu = 0.0
    cached_temp = 0.0
    cached_mem = 0.0
    cached_swap = 0.0

    try:
        while True:
            # 1. Capturar Imagem
            now_local = datetime.now(TZ_BR)
            image_array_rgb = picam2.capture_array()
            
            # 2. Executar Detecção
            detections = detect_objects_ncnn(
                image_array_rgb, ncnn_net, 
                conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD
            )
            
            # 3. Lógica de Rastreamento
            matched_detection_indices = set()
            unmatched_track_indices = set(range(len(active_tracks)))
            det_centers = [get_center(d['box']) for d in detections]
            
            for i, track in enumerate(active_tracks):
                track_center = get_center(track['box'])
                best_dist = float('inf')
                best_match_idx = -1
                for j, det_center in enumerate(det_centers):
                    if j in matched_detection_indices: continue
                    if detections[j]['class'] != track['class']: continue
                    dist = calc_distance(track_center, det_center)
                    if dist < best_dist and dist < MAX_ASSOCIATION_DIST:
                        best_dist = dist
                        best_match_idx = j
                if best_match_idx != -1:
                    track['box'] = detections[best_match_idx]['box']
                    track['frames_since_seen'] = 0 
                    matched_detection_indices.add(best_match_idx)
                    unmatched_track_indices.remove(i)
            
            for i in unmatched_track_indices:
                active_tracks[i]['frames_since_seen'] += 1
            active_tracks = [t for t in active_tracks if t['frames_since_seen'] <= MAX_FRAMES_TO_LIVE]
            
            for j, det in enumerate(detections):
                if j in matched_detection_indices: continue
                new_track = {'id': next_object_id, 'box': det['box'], 'class': det['class'], 'frames_since_seen': 0}
                active_tracks.append(new_track)
                next_object_id += 1
                if new_track['class'].lower() == 'car': total_car_count += 1
                elif new_track['class'].lower() == 'sign': total_sign_count += 1

            # 4. Desenhar Bounding Boxes
            draw_detections_on_image_cv2(image_array_rgb, detections)
            
            # 5. Calcular FPS
            loop_end_time = time.time()
            loop_time = loop_end_time - loop_start_time
            current_fps = 1.0 / loop_time
            loop_time_ms = loop_time * 1000
            loop_start_time = loop_end_time
            
            # 6. Coletar stats com "throttle" (a cada 10 frames)
            if frame_count % 10 == 0:
                cached_cpu = get_cpu_usage()
                cached_temp = get_cpu_temperature()
                cached_mem = get_memory_usage()
                cached_swap = get_swap_usage()

            # 7. Desenhar Estatísticas
            draw_system_stats_cv2(
                image_array_rgb, 
                current_fps, cached_cpu, cached_temp, cached_mem, cached_swap,
                loop_time_ms, total_car_count, total_sign_count, now_local
            )

            # 8. Enviar para a Fila de Salvamento
            final_image_np_bgr = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
            filename = os.path.join(SAVE_DIR, f"{frame_count:06d}.jpg")
            
            if not save_queue.full():
                save_queue.put({'filename': filename, 'image': final_image_np_bgr})
            else:
                print(f"[Main Thread] ATENÇÃO: Fila de salvamento cheia! Descartando frame {frame_count}.")
            
            # 9. Imprimir Status
            print(f"Frame {frame_count:04d} | "
                  f"Loop: {loop_time_ms:4.0f}ms | "
                  f"FPS: {current_fps:4.1f} | "
                  f"CPU: {cached_cpu:5.1f}% | "
                  f"RAM: {cached_mem:4.1f}% | "
                  f"Temp: {cached_temp:4.1f}°C | "
                  f"Tracks: {len(active_tracks)} | "
                  f"Queue: {save_queue.qsize()}") 

            # 10. Armazenar métricas para relatório
            all_fps_readings.append(current_fps)
            all_loop_ms_readings.append(loop_time_ms)
            
            if frame_count % 10 == 0:
                all_cpu_readings.append(cached_cpu)
                all_mem_readings.append(cached_mem)
                all_swap_readings.append(cached_swap)
                if cached_temp > 0: 
                    all_temp_readings.append(cached_temp)

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\nCaptura interrompida pelo usuário")
    finally:
        finalize_capture()

if __name__ == "__main__":
    main()