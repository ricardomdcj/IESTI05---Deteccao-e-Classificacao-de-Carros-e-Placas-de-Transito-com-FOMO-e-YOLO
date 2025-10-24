import tensorflow.lite as tflite
import numpy as np
import cv2                                  # Trocado de PIL
import os
import glob
import json
from tqdm import tqdm
import shutil 
import math                                 # Necessário para o helper

# --- Configurações (AJUSTE ESTAS VARIÁVEIS) ---

# 1. Caminho para a pasta com as imagens
IMAGE_FOLDER = './original'

# 2. Caminho para a pasta de saída das imagens processadas
# OUTPUT_IMAGE_FOLDER = './classificadas_fomo'
OUTPUT_IMAGE_FOLDER = './classificadas_fomo_320'

# 3. Caminho para o modelo TFLite
# MODEL_TFLITE = './models/ei-iesti05---trabalho-1-quantized.lite' 
MODEL_TFLITE = './models/ei-iesti05---trabalho-1-quantized-320.lite' 

# 4. Caminho para o arquivo de saída
OUTPUT_JSON = 'detection_results_fomo.json'

# 5. Outras configurações do modelo
INPUT_SIZE = None # Será definido dinamicamente ao carregar o modelo
CONFIDENCE_THRESHOLD = 0.8
MODEL_OUTPUT_CLASSES = 3 # (_background, car, sign)

# 6. Labels do modelo
labels = {
    0: 'car',
    1: 'sign'
}

# --- Variáveis Globais do TFLite ---
# (Serão preenchidas por load_tflite_model)
output_grid_w = None      
output_grid_h = None      
output_scale = 1.0        
output_zero_point = 0     
input_dtype = np.uint8    
is_int8_model = False   

# --- Funções de Inferência (Substituídas por TFLite/FOMO) ---

def load_tflite_model(model_path):
    """Carrega o modelo TFLite e define as variáveis globais de I/O."""
    global INPUT_SIZE, output_grid_h, output_grid_w
    global output_scale, output_zero_point, input_dtype, is_int8_model

    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    print(f"[TFLite] Interpretador carregado com num_threads=4.")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    INPUT_SIZE = (input_shape[2], input_shape[1]) # (Largura, Altura)
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

    return interpreter, input_details, output_details

def detect_objects_fomo_tflite(img_array_bgr, interpreter, input_details, output_details, conf_thres):
    """Executa a detecção FOMO TFLite e retorna uma lista de centroides."""
    # Esta função é adaptada para usar o tamanho da imagem de entrada
    # em vez de uma RESOLUTION global.
    
    global INPUT_SIZE, output_grid_h, output_grid_w
    global output_scale, output_zero_point, is_int8_model

    # 1. Pré-processar
    # Obtém o tamanho da imagem original para re-escalar a saída
    original_h, original_w = img_array_bgr.shape[:2] 
    
    img_resized = cv2.resize(img_array_bgr, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(img_resized, axis=0)

    # 2. Normalização (Respeitando o BGR)
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
            
            # Re-escala o centroide para o tamanho da imagem ORIGINAL
            scale_x = original_w / output_grid_w
            scale_y = original_h / output_grid_h
            
            cx = int((c + 0.5) * scale_x)
            cy = int((r + 0.5) * scale_y)
            
            detections.append({
                'class': class_name,
                'class_id': label_index, 
                'score': float(score),
                'centroid': (cx, cy) 
            })

    return detections

# --- FUNÇÃO DE DESENHO (Substituída por CV2/Centroide) ---

def draw_detections_on_image_cv2(img_array, detections):
    """Desenha detecções (centroides) no array numpy usando CV2."""
    
    draw_radius = 10 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    for det in detections:
        cx, cy = det['centroid']
        class_name = det['class']
        label = f"{class_name}: {det['score']:.2f}"
        
        if class_name.lower() == 'sign':
            color = (0, 255, 0) # Verde (BGR)
        else:
            color = (0, 0, 255) # Vermelho (BGR)
            
        # Desenha o círculo
        cv2.circle(img_array, (cx, cy), draw_radius, color, 2) 
        # Desenha a cruz
        cv2.line(img_array, (cx - 5, cy), (cx + 5, cy), color, 1)
        cv2.line(img_array, (cx, cy - 5), (cx, cy + 5), color, 1)
        
        # Desenha o texto do label (com fundo)
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        text_l = max(0, cx - (text_w // 2)) 
        text_t_baseline = max(0, cy - draw_radius - 5) 
        text_b_top = text_t_baseline - text_h - 4 
        
        cv2.rectangle(img_array, (text_l - 2, text_b_top), (text_l + text_w + 2, text_t_baseline + 4), color, -1)
        cv2.putText(img_array, label, (text_l, text_t_baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return img_array # Retorna o array modificado

# --- FUNÇÃO PRINCIPAL (Adaptada para CV2 e TFLite) ---

def main():
    print("Carregando modelo TFLite (FOMO)...")
    interpreter, input_details, output_details = load_tflite_model(MODEL_TFLITE)
    print(f"Modelo carregado. Input dinâmico: {INPUT_SIZE}")

    # Encontra todas as imagens na pasta (incluindo subpastas)
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, '**', ext), recursive=True))
    
    print(f"Encontradas {len(image_paths)} imagens em '{IMAGE_FOLDER}'.")

    # Cria a pasta de saída principal
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    print(f"Salvando imagens processadas em: '{OUTPUT_IMAGE_FOLDER}'")

    all_results = {} # Dicionário para guardar os resultados do JSON

    # Use tqdm para criar uma barra de progresso
    for img_path in tqdm(image_paths, desc="Processando imagens"):
        try:
            # Carrega a imagem com CV2 (lê em BGR, que é o que seu modelo espera)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Erro ao abrir {img_path} com cv2.")
                continue
        except Exception as e:
            print(f"Erro ao abrir {img_path}: {e}")
            continue
        
        # 1. Executa a detecção
        detections = detect_objects_fomo_tflite(img_bgr, interpreter, 
                                                input_details, output_details,
                                                conf_thres=CONFIDENCE_THRESHOLD)
        
        # 2. Define o caminho de saída
        relative_path = os.path.relpath(img_path, IMAGE_FOLDER)
        output_path = os.path.join(OUTPUT_IMAGE_FOLDER, relative_path)
        
        # 3. Garante que o subdiretório de saída exista
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        if detections:
            # 4a. Se detectou, salva no JSON e desenha na imagem
            all_results[relative_path] = detections 
            
            # Desenha os centroides na imagem (faz uma cópia para não mexer no original)
            img_with_detections = draw_detections_on_image_cv2(img_bgr.copy(), detections)
            
            # Salva a nova imagem com as detecções
            cv2.imwrite(output_path, img_with_detections)
        else:
            # 4b. Se não detectou, apenas copia a imagem original
            try:
                shutil.copy(img_path, output_path)
            except Exception as e:
                print(f"Erro ao copiar {img_path}: {e}. Tentando salvar com CV2...")
                cv2.imwrite(output_path, img_bgr) # Fallback

    # Salva todos os resultados em um único arquivo JSON
    print(f"\nProcessamento concluído. {len(all_results)} imagens tiveram detecções.")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Resultados JSON salvos em '{OUTPUT_JSON}'.")
    print(f"Todas as {len(image_paths)} imagens (processadas ou copiadas) estão em '{OUTPUT_IMAGE_FOLDER}'.")


if __name__ == "__main__":
    main()