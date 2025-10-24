import time
from picamera2 import Picamera2
from datetime import datetime
import os
import numpy as np
import signal
import sys
import cv2
import psutil
import math
from pytz import timezone
import tflite_runtime.interpreter as tflite 
import threading                           
from queue import Queue, Empty             

# --- Configurações Gerais ---
RESOLUTION = (640, 480)
TARGET_FPS = 24.0 
TARGET_FRAME_TIME = 1.0 / TARGET_FPS 
TZ_BR = timezone('America/Sao_Paulo')
folder_time = datetime.now(TZ_BR).strftime("%Y%m%d_%H%M%S")
SAVE_DIR = folder_time
os.makedirs(SAVE_DIR, exist_ok=True)
JPEG_QUALITY = 85 

# --- Configurações do Modelo TFLite (FOMO) ---
MODEL_TFLITE = './models/ei-iesti05---trabalho-1-quantized.lite' 
INPUT_SIZE = None 
CONFIDENCE_THRESHOLD = 0.5 

labels = {
    0: 'car', 
    1: 'sign' 
}
MODEL_OUTPUT_CLASSES = 3 

# --- Configurações do Rastreador/Contador ---
active_tracks = []      
next_object_id = 0      
total_car_count = 0     
total_sign_count = 0    
MAX_ASSOCIATION_DIST = 200
MAX_FRAMES_TO_LIVE = 72

# --- Fila e Sinal de Parada para a Thread de Salvamento ---
save_queue = Queue(maxsize=30) 
stop_event = threading.Event()

# --- Variáveis Globais ---
picam2 = None
tflite_interpreter = None 
input_details = None      
output_details = None     
output_grid_w = None      
output_grid_h = None      
output_scale = 1.0        
output_zero_point = 0     
input_dtype = np.uint8    
is_int8_model = False     

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
Modelo TFLite: {MODEL_TFLITE}
Input Shape: {INPUT_SIZE[0]}x{INPUT_SIZE[1]}
Limite de Confiança: {CONFIDENCE_THRESHOLD}
Limite de FPS: {TARGET_FPS} FPS 
Tolerância de Distância (Track): {MAX_ASSOCIATION_DIST} px
Tempo de Vida (Track): {MAX_FRAMES_TO_LIVE} frames
TFLite Threads: 4

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


# --- Funções de Inferência TFLite (FOMO) ---

def load_tflite_model(model_path):
    global INPUT_SIZE, input_details, output_details, output_grid_h, output_grid_w
    global output_scale, output_zero_point, input_dtype, is_int8_model

    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    print(f"[TFLite] Interpretador carregado com num_threads=4.")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    INPUT_SIZE = (input_shape[2], input_shape[1]) 
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.int8:
        is_int8_model = True
        print("[TFLite] Modelo int8 detectado (entrada/saída quantizada).")
    else:
        print("[TFLite] Modelo float32 detectado.")

    output_shape = output_details[0]['shape']
    output_grid_h = output_shape[1]
    output_grid_w = output_shape[2]
    
    if output_shape[3] != MODEL_OUTPUT_CLASSES:
        raise ValueError(f"Modelo tem {output_shape[3]} classes, mas 'MODEL_OUTPUT_CLASSES' está definido como {MODEL_OUTPUT_CLASSES}")

    if is_int8_model:
        output_quant_params = output_details[0].get('quantization_parameters', {})
        output_scale = output_quant_params.get('scales', [1.0])[0]
        output_zero_point = output_quant_params.get('zero_points', [0])[0]

    return interpreter

def detect_objects_fomo_tflite(img_array_rgb, interpreter, conf_thres):
    global INPUT_SIZE, input_details, output_details, output_grid_h, output_grid_w
    global output_scale, output_zero_point, is_int8_model

    # 1. Pré-processar
    img_resized = cv2.resize(img_array_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(img_resized, axis=0)

    # 2. Normalização
    if is_int8_model:
        input_data = input_data.astype(np.float32) - 128.0
        input_data = input_data.astype(np.int8)
    else:
        input_data = (input_data.astype(np.float32) / 127.5) - 1.0

    # 3. Inferência
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 4. Obter Saída
    heatmap_raw = interpreter.get_tensor(output_details[0]['index'])
    heatmap = np.squeeze(heatmap_raw) 

    # 5. De-quantizar
    if is_int8_model:
        heatmap = (heatmap.astype(np.float32) - output_zero_point) * output_scale

    detections = []

    # 6. Encontrar Picos
    for class_id_model in range(1, MODEL_OUTPUT_CLASSES):
        class_heatmap = heatmap[:, :, class_id_model]
        locations = np.where(class_heatmap >= conf_thres)
        
        label_index = class_id_model - 1
        
        if label_index not in labels:
            continue
            
        class_name = labels[label_index]

        for r, c in zip(locations[0], locations[1]):
            score = class_heatmap[r, c]
            
            scale_x = RESOLUTION[0] / output_grid_w
            scale_y = RESOLUTION[1] / output_grid_h
            
            cx = int((c + 0.5) * scale_x)
            cy = int((r + 0.5) * scale_y)
            
            detections.append({
                'class': class_name,
                'class_id': label_index, 
                'score': float(score),
                'centroid': (cx, cy) 
            })

    return detections

# --- Funções de Desenho (CV2) ---

def draw_detections_on_image_cv2(img_array, detections):
    draw_radius = 10 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    for det in detections:
        cx, cy = det['centroid']
        class_name = det['class']
        label = f"{class_name}: {det['score']:.2f}"
        
        if class_name.lower() == 'sign':
            color = (0, 255, 0) # Verde
        else:
            color = (0, 0, 255) # Vermelho
            
        cv2.circle(img_array, (cx, cy), draw_radius, color, 2) 
        cv2.line(img_array, (cx - 5, cy), (cx + 5, cy), color, 1)
        cv2.line(img_array, (cx, cy - 5), (cx, cy + 5), color, 1)
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_l = max(0, cx - (text_w // 2)) 
        text_t_baseline = max(0, cy - draw_radius - 5) 
        text_b_top = text_t_baseline - text_h - 4 
        
        cv2.rectangle(img_array, (text_l - 2, text_b_top), (text_l + text_w + 2, text_t_baseline + 4), color, -1)
        cv2.putText(img_array, label, (text_l, text_t_baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return img_array 

def draw_system_stats_cv2(img_array, fps, cpu, temp, mem, swap, loop_ms, car_count, sign_count, now_timestamp):
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
    global picam2, tflite_interpreter, frame_count, loop_start_time
    global active_tracks, next_object_id, total_car_count, total_sign_count
    global global_start_time, all_fps_readings, all_cpu_readings, all_temp_readings
    global all_mem_readings, all_swap_readings, all_loop_ms_readings
    global INPUT_SIZE 
    
    print("Carregando modelo TFLite (FOMO)...")
    tflite_interpreter = load_tflite_model(MODEL_TFLITE)
    print(f"Modelo carregado. Input: {INPUT_SIZE[0]}x{INPUT_SIZE[1]}. Saída grid: {output_grid_w}x{output_grid_h}")

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
    
    print("Iniciando thread de salvamento em segundo plano...")
    saver_thread = threading.Thread(target=save_worker, daemon=True)
    saver_thread.start()

    print(f"\nPasta de destino: {SAVE_DIR}/")
    print(f"Iniciando captura... (Limite de {TARGET_FPS} FPS - Pressione Ctrl+C para parar)\n")
    print("-" * 80)
    
    psutil.cpu_percent(interval=None) 
    time.sleep(0.5)
    
    global_start_time = datetime.now(TZ_BR)
    loop_start_time = time.time()
    
    cached_cpu = 0.0
    cached_temp = 0.0
    cached_mem = 0.0
    cached_swap = 0.0

    try:
        while True:
            # 1. Capturar Imagem
            now_local = datetime.now(TZ_BR)
            image_array_rgb = picam2.capture_array() # Formato: Numpy RGB
            
            # 2. Executar Detecção (FOMO)
            detections = detect_objects_fomo_tflite(
                image_array_rgb.copy(), 
                tflite_interpreter, 
                conf_thres=CONFIDENCE_THRESHOLD
            )
            
            # 3. Lógica de Rastreamento
            matched_detection_indices = set()
            unmatched_track_indices = set(range(len(active_tracks)))
            det_centers = [d['centroid'] for d in detections]
            
            for i, track in enumerate(active_tracks):
                track_center = track['centroid'] 
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
                    track['centroid'] = detections[best_match_idx]['centroid'] 
                    track['frames_since_seen'] = 0 
                    matched_detection_indices.add(best_match_idx)
                    unmatched_track_indices.remove(i)
            
            for i in unmatched_track_indices:
                active_tracks[i]['frames_since_seen'] += 1

            active_tracks = [t for t in active_tracks if t['frames_since_seen'] <= MAX_FRAMES_TO_LIVE]
            
            for j, det in enumerate(detections):
                if j in matched_detection_indices: continue
                new_track = {
                    'id': next_object_id,
                    'centroid': det['centroid'], 
                    'class': det['class'],
                    'frames_since_seen': 0
                }
                active_tracks.append(new_track)
                next_object_id += 1
                if new_track['class'].lower() == 'car': total_car_count += 1
                elif new_track['class'].lower() == 'sign': total_sign_count += 1

            # 4. Desenhar Detecções
            draw_detections_on_image_cv2(image_array_rgb, detections)
            
            # 5. Calcular FPS e Limitar (Target FPS)
            loop_end_time = time.time()
            processing_time = loop_end_time - loop_start_time # Tempo só de processamento
            
            # Calcula o tempo de espera
            sleep_time = TARGET_FRAME_TIME - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time) # Dorme para atingir o TARGET_FRAME_TIME
            
            # Recalcula o tempo total do loop após o sleep
            final_loop_end_time = time.time()
            loop_time = final_loop_end_time - loop_start_time # Tempo total (process + sleep)
            
            current_fps = 1.0 / loop_time
            loop_time_ms = loop_time * 1000
            loop_start_time = final_loop_end_time # Define o início do PRÓXIMO loop
            
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