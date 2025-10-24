import cv2
import os
from pathlib import Path
import argparse # Vamos remover o uso dele, mas pode deixar o import
from tqdm import tqdm
import re # Importado para natural_sort_key

# ==================== CONFIGURAÇÕES ====================
# As pastas de entrada e saída serão definidas interativamente

# FPS do vídeo (frames por segundo)
VIDEO_FPS = 24

# Qualidade do vídeo (0-51, onde 0 é melhor qualidade e maior arquivo)
CRF_QUALITY = 23

# Rotacionar imagens 180 graus? (True ou False)
ROTATE_180 = False  

# Codec de vídeo
CODEC = 'mp4v'

# ========================================================


def natural_sort_key(s):
    """
    Chave de ordenação natural para ordenar corretamente arquivos numerados
    Exemplo: 000001.jpg, 000002.jpg, ..., 000010.jpg
    """
    # import re (já importado no topo)
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', str(s))]


def get_image_files(folder):
    """Obtém lista de arquivos de imagem ordenados naturalmente"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")

    # Listar todos os arquivos de imagem
    image_files = [f for f in folder_path.iterdir() 
                   if f.suffix.lower() in extensions]

    if not image_files:
        raise ValueError(f"Nenhuma imagem encontrada em: {folder}")

    # Ordenar naturalmente
    image_files.sort(key=natural_sort_key)

    return image_files


def create_video_from_images(input_folder, output_video, fps=24, 
                             rotate_180=False, codec='avc1', crf=23):
    """
    Cria um vídeo a partir de uma sequência de imagens

    Parâmetros:
        input_folder: Pasta contendo as imagens
        output_video: Nome do arquivo de vídeo de saída
        fps: Frames por segundo do vídeo
        rotate_180: Se True, rotaciona as imagens 180 graus
        codec: Codec de vídeo ('mp4v', 'avc1', 'H264', etc.)
        crf: Qualidade do vídeo (apenas para alguns codecs)
    """

    print("="*70)
    print("CRIANDO VÍDEO A PARTIR DE IMAGENS")
    print("="*70)

    # Obter lista de imagens
    print(f"\nBuscando imagens em: {input_folder}")
    image_files = get_image_files(input_folder)
    total_images = len(image_files)
    print(f"✓ Encontradas {total_images} imagens")

    # Ler primeira imagem para obter dimensões
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_files[0]}")

    # Se precisar rotacionar, ajustar dimensões
    if rotate_180:
        first_image = cv2.rotate(first_image, cv2.ROTATE_180)
        print("✓ Rotação de 180° será aplicada")

    height, width, layers = first_image.shape
    frame_size = (width, height)

    print(f"\nResolução: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Codec: {codec}")
    print(f"Duração estimada: {total_images/fps:.2f} segundos")

    # Configurar VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    if not video_writer.isOpened():
        raise RuntimeError(f"Erro ao criar VideoWriter com codec '{codec}'.\n"
                          f"Tente usar 'mp4v' como codec alternativo.")

    # Processar todas as imagens
    print(f"\nProcessando imagens...")
    for img_path in tqdm(image_files, desc="Criando vídeo", unit="frame"):
        # Ler imagem
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"\n⚠ Aviso: Não foi possível ler {img_path.name}, pulando...")
            continue

        # Rotacionar se necessário
        if rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Verificar se as dimensões correspondem
        if frame.shape[:2] != (height, width):
            print(f"\n⚠ Aviso: Imagem {img_path.name} tem dimensão diferente, "
                  f"redimensionando...")
            frame = cv2.resize(frame, frame_size)

        # Escrever frame no vídeo
        video_writer.write(frame)

    # Liberar recursos
    video_writer.release()

    # Verificar se o arquivo foi criado
    output_path = Path(output_video)
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*70}")
        print(f"✓ VÍDEO CRIADO COM SUCESSO!")
        print(f"{'='*70}")
        print(f"Arquivo: {output_video}")
        print(f"Tamanho: {file_size_mb:.2f} MB")
        print(f"Frames: {total_images}")
        print(f"FPS: {fps}")
        print(f"Duração: {total_images/fps:.2f} segundos")
        print(f"{'='*70}")
    else:
        raise RuntimeError("Erro: O arquivo de vídeo não foi criado")


def main():
    """Função principal interativa"""
    
    print("="*50)
    print(" Criador de Vídeo a partir de Imagens")
    print("="*50)
    print("Selecione a fonte das imagens:")
    print("  [1] Pasta './original'")
    print("  [2] Pasta './classificadas_fomo'")
    print("  [3] Pasta './classificadas_yolo'")
    print("  [4] Pasta './classificadas_fomo_320'")
    print("\n  [Qualquer outra tecla] Sair")
    print("="*50)

    choice = input("Digite sua escolha (1, 2, 3 ou 4): ")

    input_folder = None
    output_video = None

    if choice == '1':
        input_folder = "./original"
        output_video = "output_video_original.mp4"
    elif choice == '2':
        input_folder = "./classificadas_fomo"
        output_video = "output_video_classificadas_fomo.mp4"
    elif choice == '3':
        input_folder = "./classificadas_yolo"
        output_video = "output_video_classificadas_yolo.mp4"
    elif choice == '4':
        input_folder = "./classificadas_fomo_320"
        output_video = "output_video_classificadas_fomo_320.mp4"
    else:
        print("Opção inválida. Saindo...")
        return 1
    
    print(f"\nOpção selecionada: {choice}")
    print(f"  - Pasta de Entrada: {input_folder}")
    print(f"  - Vídeo de Saída:   {output_video}")
    print(f"  - FPS:              {VIDEO_FPS}")
    print(f"  - Codec:            {CODEC}")
    print(f"  - Rotação 180:      {ROTATE_180}")
    print("-" * 50)


    try:
        # Chama a função de criação de vídeo com as opções selecionadas
        # e as configurações globais
        create_video_from_images(
            input_folder=input_folder,
            output_video=output_video,
            fps=VIDEO_FPS,
            rotate_180=ROTATE_180,
            codec=CODEC
            # Nota: CRF_QUALITY é usado por padrão dentro da função
        )

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        print("\nDicas:")
        print(f"1. Verifique se a pasta '{input_folder}' existe")
        print("2. Verifique se há imagens na pasta")
        print("3. Tente usar codec 'mp4v' se 'avc1' não funcionar (mude a variável global CODEC)")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())