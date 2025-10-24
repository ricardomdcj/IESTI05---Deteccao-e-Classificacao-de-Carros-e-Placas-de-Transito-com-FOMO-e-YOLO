import ncnn
import numpy as np
from PIL import Image, ImageDraw, ImageFont ### NOVO: Importa ImageDraw e ImageFont ###
import os
import glob
import json
from tqdm import tqdm
import shutil ### NOVO: Para copiar arquivos rapidamente ###

# --- Configurações (AJUSTE ESTAS VARIÁVEIS) ---

# 1. Caminho para a pasta com as 15.000 imagens
IMAGE_FOLDER = './original'

# 2. ### NOVO: Caminho para a pasta de saída das imagens com caixas ###
OUTPUT_IMAGE_FOLDER = './classificadas_yolo'

# 3. Caminho para os modelos NCNN
MODEL_PARAM = './models/best_ncnn_model/model.ncnn.param'
MODEL_BIN   = './models/best_ncnn_model/model.ncnn.bin'

# 4. Caminho para o arquivo de saída
OUTPUT_JSON = 'detection_results.json'

# 5. Outras configurações do modelo
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# 6. Ajuste os labels para o seu modelo
# (Copie exatamente do seu script original)
labels = {
    0: 'car',
    1: 'sign'
}

# --- Funções de Inferência (Sem alterações) ---

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

def load_ncnn_model():
    ncnn_net = ncnn.Net()
    ncnn_net.load_param(MODEL_PARAM)
    ncnn_net.load_model(MODEL_BIN)
    return ncnn_net

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
    
    try:
        if not hasattr(detect_objects_ncnn, 'input_name'):
            detect_objects_ncnn.input_name = try_input(ex, mat_in)
        else:
            ex.input(detect_objects_ncnn.input_name, mat_in)
            
        if not hasattr(detect_objects_ncnn, 'output_name'):
             _, detect_objects_ncnn.output_name = extract_any(ex)
             
    except RuntimeError as e:
        print(f"Erro ao configurar extractor: {e}")
        if hasattr(detect_objects_ncnn, 'input_name'): 
            del detect_objects_ncnn.input_name
        if hasattr(detect_objects_ncnn, 'output_name'): 
            del detect_objects_ncnn.output_name
        return []

    ret, out_mat = ex.extract(detect_objects_ncnn.output_name)
    if ret != 0:
        return []

    if getattr(out_mat, 'elemsize', 0) not in (1, 2, 4):
        raise RuntimeError(f"Output elemsize inválido: {getattr(out_mat, 'elemsize', None)} do blob {detect_objects_ncnn.output_name}")

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

# --- ### FUNÇÃO DE DESENHO MODIFICADA ### ---

def draw_detections_on_image(img_pil, detections):
    """Desenha as detecções em uma cópia da imagem PIL."""
    img_with_boxes = img_pil.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    try:
        # Tenta carregar uma fonte comum (presente no Windows/macOS)
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        # Se falhar, usa a fonte padrão (pode ser pequena)
        print("Fonte Arial não encontrada, usando fonte padrão.")
        font = ImageFont.load_default()

    for det in detections:
        l, t, r, b = det['box']
        class_name = det['class'] # Pega o nome da classe
        label = f"{class_name}: {det['score']:.2f}"

        # --- ### LÓGICA DA COR ### ---
        # Define a cor com base no nome da classe
        if class_name.lower() == 'sign': # Verifica se é 'sign' (ignorando maiúsculas)
            box_color = "green"
        else:
            box_color = "red"
        # --- ### FIM DA LÓGICA DA COR ### ---

        # Desenha o retângulo (usando a nova 'box_color')
        draw.rectangle([l, t, r, b], outline=box_color, width=3)
        
        # Calcula o tamanho do texto para desenhar um fundo
        try:
            # textbbox é a forma moderna
            text_box = draw.textbbox((l, t), label, font=font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
        except AttributeError:
            # textsize é a forma antiga (fallback)
            text_w, text_h = draw.textsize(label, font=font)

        # Posição do texto (acima da caixa)
        text_t = max(0, t - text_h - 2) # -2 para um pequeno padding
        
        # Desenha o fundo sólido para o texto (usando a nova 'box_color')
        draw.rectangle([l, text_t, l + text_w + 2, text_t + text_h + 2], fill=box_color)
        # Desenha o texto
        draw.text((l + 1, text_t + 1), label, font=font, fill="white")
            
    return img_with_boxes

# --- ### FUNÇÃO PRINCIPAL (MODIFICADA) ### ---

def main():
    print("Carregando modelo NCNN...")
    net = load_ncnn_model()
    print("Modelo carregado.")

    # Encontra todas as imagens na pasta (incluindo subpastas)
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, '**', ext), recursive=True))
    
    print(f"Encontradas {len(image_paths)} imagens em '{IMAGE_FOLDER}'.")

    # ### NOVO: Cria a pasta de saída principal ###
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    print(f"Salvando imagens processadas em: '{OUTPUT_IMAGE_FOLDER}'")

    all_results = {} # Dicionário para guardar os resultados do JSON

    # Use tqdm para criar uma barra de progresso
    for img_path in tqdm(image_paths, desc="Processando imagens"):
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Erro ao abrir {img_path}: {e}")
            continue
        
        # 1. Executa a detecção
        detections = detect_objects_ncnn(img_pil, net, 
                                         conf_thres=CONFIDENCE_THRESHOLD, 
                                         iou_thres=IOU_THRESHOLD)
        
        # 2. ### NOVO: Define o caminho de saída ###
        # Pega o caminho relativo (ex: 'subpasta/imagem.jpg')
        relative_path = os.path.relpath(img_path, IMAGE_FOLDER)
        # Monta o caminho de saída completo
        output_path = os.path.join(OUTPUT_IMAGE_FOLDER, relative_path)
        
        # 3. ### NOVO: Garante que o subdiretório de saída exista ###
        # (ex: 'caminho/para/imagens_processadas/subpasta')
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        if detections:
            # 4a. ### NOVO: Se detectou, salva no JSON e desenha na imagem ###
            all_results[relative_path] = detections 
            
            img_with_boxes = draw_detections_on_image(img_pil, detections)
            
            # Salva a nova imagem com as caixas
            img_with_boxes.save(output_path)
        else:
            # 4b. ### NOVO: Se não detectou, apenas copia a imagem original ###
            # Isso é mais rápido do que recodificar a imagem com PIL
            try:
                shutil.copy(img_path, output_path)
            except Exception as e:
                print(f"Erro ao copiar {img_path}: {e}. Tentando salvar com PIL...")
                img_pil.save(output_path) # Fallback

    # Salva todos os resultados em um único arquivo JSON (como antes)
    print(f"\nProcessamento concluído. {len(all_results)} imagens tiveram detecções.")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Resultados JSON salvos em '{OUTPUT_JSON}'.")
    print(f"Todas as {len(image_paths)} imagens (processadas ou copiadas) estão em '{OUTPUT_IMAGE_FOLDER}'.")


if __name__ == "__main__":
    main()