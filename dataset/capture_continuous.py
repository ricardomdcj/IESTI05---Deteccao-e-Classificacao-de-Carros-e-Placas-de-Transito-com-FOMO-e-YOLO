import time
from picamera2 import Picamera2
from datetime import datetime
import os
from PIL import Image
import numpy as np
import signal
import sys
import cv2

# Configurações
TARGET_FPS = 24
TARGET_FRAME_TIME = 1.0 / TARGET_FPS  # ~41.67 ms por frame
RESOLUTION = (640, 480)  # 480p para melhor performance na Zero 2W

# Criar pasta com data e hora atual
folder_time = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = folder_time
os.makedirs(SAVE_DIR, exist_ok=True)

# Variáveis globais para métricas
frame_count = 0
total_capture_time = 0
total_save_time = 0
slack_times = []
dropped_frames = 0
start_time = None
picam2 = None

def signal_handler(sig, frame):
    """Handler para capturar Ctrl+C e finalizar graciosamente"""
    print("\n\nCaptura interrompida pelo usuário (Ctrl+C)")
    finalize_capture()
    sys.exit(0)

def save_metrics_to_file():
    """Salva todas as métricas em um arquivo de texto"""
    if frame_count > 0:
        total_time = time.time() - start_time
        actual_fps = frame_count / total_time
        avg_capture_time = (total_capture_time / frame_count) * 1000
        avg_save_time = (total_save_time / frame_count) * 1000
        avg_slack = (sum(slack_times) / len(slack_times)) * 1000
        min_slack = min(slack_times) * 1000
        max_slack = max(slack_times) * 1000

        # Calcular desvio padrão do slack
        slack_ms = [s * 1000 for s in slack_times]
        std_slack = np.std(slack_ms)

        # Nome do arquivo de métricas
        metrics_filename = os.path.join(SAVE_DIR, "metricas_captura.txt")

        with open(metrics_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO DE CAPTURA DE IMAGENS\n")
            f.write("=" * 80 + "\n\n")

            # Informações gerais
            f.write("INFORMAÇÕES GERAIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Data e hora da captura: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Diretório de salvamento: {SAVE_DIR}/\n")
            f.write(f"Resolução: {RESOLUTION[0]}x{RESOLUTION[1]}\n")
            f.write(f"Qualidade JPEG: 85\n\n")

            # Estatísticas de captura
            f.write("ESTATÍSTICAS DE CAPTURA\n")
            f.write("-" * 80 + "\n")
            f.write(f"Frames capturados: {frame_count}\n")
            f.write(f"Tempo total de execução: {total_time:.2f} s\n")
            f.write(f"FPS alvo: {TARGET_FPS:.2f}\n")
            f.write(f"FPS real alcançado: {actual_fps:.2f}\n")
            f.write(f"Eficiência: {(actual_fps/TARGET_FPS)*100:.1f}%\n\n")

            # Tempos de processamento
            f.write("TEMPOS DE PROCESSAMENTO\n")
            f.write("-" * 80 + "\n")
            f.write(f"Tempo alvo por frame: {TARGET_FRAME_TIME*1000:.2f} ms\n")
            f.write(f"Tempo médio de captura: {avg_capture_time:.2f} ms\n")
            f.write(f"Tempo médio de salvamento: {avg_save_time:.2f} ms\n")
            f.write(f"Tempo médio total: {(avg_capture_time + avg_save_time):.2f} ms\n\n")

            # Análise de slack
            f.write("ANÁLISE DE SLACK (Tempo Disponível)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Slack médio: {avg_slack:+.2f} ms\n")
            f.write(f"Slack mínimo: {min_slack:+.2f} ms\n")
            f.write(f"Slack máximo: {max_slack:+.2f} ms\n")
            f.write(f"Desvio padrão do slack: {std_slack:.2f} ms\n\n")

            # Frames atrasados
            f.write("FRAMES ATRASADOS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total de frames atrasados: {dropped_frames}\n")
            f.write(f"Percentual de frames atrasados: {(dropped_frames/frame_count)*100:.1f}%\n")
            f.write(f"Frames dentro do tempo: {frame_count - dropped_frames}\n")
            f.write(f"Percentual de sucesso: {((frame_count - dropped_frames)/frame_count)*100:.1f}%\n\n")

            # Resumo de desempenho
            f.write("RESUMO DE DESEMPENHO\n")
            f.write("-" * 80 + "\n")
            if actual_fps >= TARGET_FPS * 0.95:
                desempenho = "EXCELENTE"
            elif actual_fps >= TARGET_FPS * 0.85:
                desempenho = "BOM"
            elif actual_fps >= TARGET_FPS * 0.70:
                desempenho = "REGULAR"
            else:
                desempenho = "RUIM"
            f.write(f"Avaliação de desempenho: {desempenho}\n")
            f.write(f"Taxa de captura real: {actual_fps:.2f} fps\n")
            f.write(f"Margem de slack disponível: {avg_slack:+.2f} ms\n\n")

            # Recomendações
            f.write("RECOMENDAÇÕES\n")
            f.write("-" * 80 + "\n")
            if avg_slack < -5:
                f.write("⚠ Sistema sobrecarregado: Considere reduzir FPS ou resolução\n")
            elif avg_slack < 0:
                f.write("⚠ Sistema no limite: Monitorar desempenho em capturas longas\n")
            elif avg_slack < 5:
                f.write("✓ Sistema operando adequadamente com pouca margem\n")
            else:
                f.write("✓ Sistema operando com boa margem de processamento\n")

            if dropped_frames > frame_count * 0.1:
                f.write("⚠ Alto percentual de frames atrasados detectado\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"\n✓ Métricas salvas em: {metrics_filename}")
        return metrics_filename
    else:
        print("\nNenhum frame capturado. Arquivo de métricas não criado.")
        return None

def finalize_capture():
    """Finaliza a captura e salva métricas"""
    global picam2

    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
        except:
            pass

    # Imprimir estatísticas no terminal
    if frame_count > 0:
        total_time = time.time() - start_time
        actual_fps = frame_count / total_time
        avg_capture_time = (total_capture_time / frame_count) * 1000
        avg_save_time = (total_save_time / frame_count) * 1000
        avg_slack = (sum(slack_times) / len(slack_times)) * 1000
        min_slack = min(slack_times) * 1000
        max_slack = max(slack_times) * 1000

        print("\n" + "=" * 80)
        print("ESTATÍSTICAS FINAIS")
        print("=" * 80)
        print(f"Frames capturados: {frame_count}")
        print(f"Tempo total: {total_time:.2f} s")
        print(f"FPS alvo: {TARGET_FPS:.2f}")
        print(f"FPS real: {actual_fps:.2f}")
        print(f"Eficiência: {(actual_fps/TARGET_FPS)*100:.1f}%")
        print(f"\nTempo médio de captura: {avg_capture_time:.2f} ms")
        print(f"Tempo médio de salvamento: {avg_save_time:.2f} ms")
        print(f"Tempo alvo por frame: {TARGET_FRAME_TIME*1000:.2f} ms")
        print(f"\nSlack médio: {avg_slack:+.2f} ms")
        print(f"Slack mínimo: {min_slack:+.2f} ms")
        print(f"Slack máximo: {max_slack:+.2f} ms")
        print(f"\nFrames atrasados: {dropped_frames} ({(dropped_frames/frame_count)*100:.1f}%)")
        print(f"\nImagens salvas em: {SAVE_DIR}/")
        print("=" * 80)

        # Salvar métricas em arquivo
        save_metrics_to_file()
    else:
        print("\nNenhum frame foi capturado.")

# Registrar handler para Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Inicializar câmera
picam2 = Picamera2()

# Configuração otimizada para captura rápida
config = picam2.create_still_configuration(
    main={"size": RESOLUTION, "format": "BGR888"},
    buffer_count=6,
    controls={
        'FrameRate': TARGET_FPS,
        'NoiseReductionMode': 0
    }
)

print(f"Configuração: {config}")
picam2.configure(config)
picam2.start()

# Aguardar estabilização da câmera
print("Aguardando estabilização da câmera...")
time.sleep(2)

print(f"\nPasta de destino: {SAVE_DIR}/")
print(f"Capturando a {TARGET_FPS} fps (Pressione Ctrl+C para parar)")
print(f"Tempo alvo por frame: {TARGET_FRAME_TIME*1000:.2f} ms\n")
print("-" * 80)

start_time = time.time()
next_capture_time = start_time

try:
    while True:
        loop_start = time.time()

        # Capturar imagem como array
        capture_start = time.time()
        image_array = picam2.capture_array()
        capture_time = time.time() - capture_start
        total_capture_time += capture_time

        # Salvar imagem usando PIL com correção de cor RGB->BGR
        save_start = time.time()
        filename = os.path.join(SAVE_DIR, f"{frame_count:06d}.jpg")

        # CORREÇÃO: Converter RGB para BGR invertendo canais
        # Picamera2 captura em RGB, mas precisamos inverter para cores corretas
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Salvar usando OpenCV (que trabalha nativamente com BGR)
        cv2.imwrite(filename, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        save_time = time.time() - save_start
        total_save_time += save_time

        # Calcular tempo total do loop
        loop_time = time.time() - loop_start

        # Calcular slack (tempo disponível)
        slack = TARGET_FRAME_TIME - loop_time
        slack_times.append(slack)

        # Verificar se está atrasado
        if slack < 0:
            dropped_frames += 1
            status = "ATRASADO"
        else:
            status = "OK"

        frame_count += 1

        # Imprimir métricas a cada 24 frames (1 segundo)
        if frame_count % TARGET_FPS == 0:
            avg_slack = sum(slack_times[-TARGET_FPS:]) / TARGET_FPS
            elapsed = time.time() - start_time
            print(f"Frame {frame_count:04d} | "
                  f"Tempo: {elapsed:.1f}s | "
                  f"Captura: {capture_time*1000:.2f}ms | "
                  f"Salvar: {save_time*1000:.2f}ms | "
                  f"Total: {loop_time*1000:.2f}ms | "
                  f"Slack: {slack*1000:+.2f}ms | "
                  f"Slack médio: {avg_slack*1000:+.2f}ms | "
                  f"{status}")

        # Aguardar para manter o framerate (se houver slack)
        next_capture_time += TARGET_FRAME_TIME
        sleep_time = next_capture_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n\nCaptura interrompida pelo usuário (Ctrl+C)")

finally:
    finalize_capture()
