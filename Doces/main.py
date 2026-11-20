"""Execute sem argumentos para abrir a interface; use --help para o terminal."""
import argparse
from pathlib import Path
from doces_ferramentas.vision import load_config
from doces_ferramentas.processing import process_image, process_video, IMAGE_SUFFIXES, VIDEO_SUFFIXES


def main():
    parser = argparse.ArgumentParser(description="Contagem de doces e ferramentas com HSV e rastreamento")
    parser.add_argument("--entrada", type=Path, help="Uma foto ou um vídeo")
    parser.add_argument("--saida", type=Path, default=Path("resultados"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--analise-fps", type=float, help="Frequência de análise do vídeo (padrão: 15)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.analise_fps is not None:
        if args.analise_fps <= 0:
            parser.error("--analise-fps deve ser positivo")
        config["analysis_fps"] = args.analise_fps
    if args.entrada:
        path = args.entrada
        if path.suffix.lower() in IMAGE_SUFFIXES:
            output_path = process_image(path, args.saida, config)
            print(f"Imagem processada: {output_path}")
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            last_status = [None]
            def report_progress(message, _):
                if message != last_status[0]:
                    print(message, flush=True)
                    last_status[0] = message
            output_path = process_video(path, args.saida, config, progress=report_progress)
            print(f"Vídeo processado: {output_path}")
        else:
            parser.error(f"Formato não suportado: {path.suffix}")
    else:
        from doces_ferramentas.gui import launch
        launch()


if __name__ == "__main__":
    main()
