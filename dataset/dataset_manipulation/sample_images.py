import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

# ==================== CONFIGURAÇÕES ====================
# Diretório de origem com as imagens
INPUT_FOLDER = "original"  # Pasta atual por padrão

# Nome da pasta de destino (será criada dentro do INPUT_FOLDER)
OUTPUT_FOLDER = "sampled_images"

# Intervalo: copiar 1 a cada N imagens
SAMPLE_INTERVAL = 12  # Copia 1 a cada 4 imagens

# ========================================================


def natural_sort_key(s):
    """
    Chave de ordenação natural para ordenar corretamente arquivos numerados
    Exemplo: 000001.jpg, 000002.jpg, ..., 000010.jpg
    """
    import re
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', str(s))]


def get_image_files(folder):
    """Obtém lista de arquivos de imagem ordenados naturalmente"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    folder_path = Path(folder).resolve()

    if not folder_path.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder_path}")

    # Listar todos os arquivos de imagem
    image_files = [f for f in folder_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in extensions]

    if not image_files:
        raise ValueError(f"Nenhuma imagem encontrada em: {folder_path}")

    # Ordenar naturalmente
    image_files.sort(key=natural_sort_key)

    return image_files


def sample_images(input_folder, output_folder, interval=4):
    """
    Copia uma imagem a cada 'interval' imagens

    Parâmetros:
        input_folder: Pasta contendo as imagens originais
        output_folder: Nome da pasta de destino
        interval: Intervalo de amostragem (copiar 1 a cada N)
    """

    print("="*70)
    print("AMOSTRAGEM DE IMAGENS")
    print("="*70)

    # Obter lista de imagens
    print(f"\nBuscando imagens em: {input_folder}")
    image_files = get_image_files(input_folder)
    total_images = len(image_files)
    print(f"✓ Encontradas {total_images} imagens")

    # Calcular quantas imagens serão copiadas
    num_sampled = (total_images + interval - 1) // interval
    print(f"✓ Serão copiadas {num_sampled} imagens (1 a cada {interval})")

    # Criar pasta de destino
    input_path = Path(input_folder).resolve()
    output_path = Path(output_folder).resolve()

    if output_path.exists():
        response = input(f"\n⚠ A pasta '{output_folder}' já existe. Sobrescrever? (s/n): ")
        if response.lower() != 's':
            print("Operação cancelada.")
            return
        print("Limpando pasta existente...")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Pasta criada: {output_path}")

    # Copiar imagens
    print(f"\nCopiando imagens com intervalo de {interval}...")
    copied_count = 0

    for idx, img_path in enumerate(tqdm(image_files, desc="Processando", unit="img")):
        # Copiar apenas se o índice for múltiplo do intervalo
        if idx % interval == 0:
            # Manter o nome original do arquivo
            dest_path = output_path / img_path.name
            shutil.copy2(img_path, dest_path)
            copied_count += 1

    # Estatísticas finais
    print(f"\n{'='*70}")
    print(f"✓ AMOSTRAGEM CONCLUÍDA!")
    print(f"{'='*70}")
    print(f"Imagens originais: {total_images}")
    print(f"Imagens copiadas: {copied_count}")
    print(f"Taxa de amostragem: 1/{interval} ({(copied_count/total_images)*100:.1f}%)")
    print(f"Pasta de destino: {output_path}/")
    print(f"{'='*70}")

    # Mostrar exemplos dos arquivos copiados
    sampled_files = sorted(output_path.iterdir(), key=natural_sort_key)
    if len(sampled_files) <= 10:
        print(f"\nArquivos copiados:")
        for f in sampled_files:
            print(f"  - {f.name}")
    else:
        print(f"\nPrimeiros arquivos copiados:")
        for f in sampled_files[:5]:
            print(f"  - {f.name}")
        print(f"  ...")
        print(f"Últimos arquivos copiados:")
        for f in sampled_files[-5:]:
            print(f"  - {f.name}")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Copia uma imagem a cada N imagens para uma nova pasta')

    parser.add_argument('-i', '--input', type=str, default=INPUT_FOLDER,
                       help=f'Pasta com imagens (padrão: pasta atual)')

    parser.add_argument('-o', '--output', type=str, default=OUTPUT_FOLDER,
                       help=f'Nome da pasta de destino (padrão: {OUTPUT_FOLDER})')

    parser.add_argument('-n', '--interval', type=int, default=SAMPLE_INTERVAL,
                       help=f'Intervalo de amostragem (padrão: {SAMPLE_INTERVAL})')

    args = parser.parse_args()

    try:
        sample_images(
            input_folder=args.input,
            output_folder=args.output,
            interval=args.interval
        )

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        print("\nDicas:")
        print("1. Verifique se a pasta de imagens existe")
        print("2. Certifique-se de que há imagens na pasta")
        print("3. Verifique as permissões de escrita")
        print("\nExemplo de uso:")
        print(f"  python {Path(__file__).name} -i pasta_imagens -n 4")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
