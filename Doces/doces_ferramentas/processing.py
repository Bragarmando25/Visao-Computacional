"""Coordena a leitura, a análise e a gravação dos resultados visuais.

Fotos geram resultado.jpg. Vídeos geram resultado.mp4 e
ultimo_quadro.jpg, preservando os quadros e a duração do trecho processado.
"""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from .vision import Detector, hand_present, read_image, write_image
from .tracking import Tracker
from .render import annotate

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def process_image(path, output_root, config, progress=None):
    """Lê uma foto, identifica os objetos e salva a imagem anotada."""
    path = Path(path)
    dest = Path(output_root) / path.stem
    dest.mkdir(parents=True, exist_ok=True)
    
    frame = read_image(path)
    detections = Detector(config).detect(frame)
    annotated = annotate(frame, detections)
    output_path = dest / "resultado.jpg"
    write_image(output_path, annotated)
    if progress:
        progress(f"Concluído: {path.name}", annotated)
    return output_path


def process_video(path, output_root, config, progress=None, stop=None):
    """Processa os quadros e salva o vídeo anotado e seu último quadro.

    O callback progress atualiza a prévia da interface. Se stop indicar uma
    interrupção, o vídeo e o último quadro do trecho concluído são preservados.
    """
    path = Path(path)
    label = path.stem
    dest = Path(output_root) / label
    dest.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {path}")
    
    #quantidade de quadros por segundo 
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30
        
    #numero total de quadros informados pelo arquivo
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Vídeos preservam todos os quadros e a duração; a análise é amostrada.
    # De quantos em quantos  quadros do video o sistema vai detectar objetos, com analysis_fps = 15, ele analisa 15 quadros por segundo de um total de 30
    stride = max(1, round(source_fps / config["analysis_fps"]))
    
    #pega as informações necessarias do JSON, como max_dimension, video_max_dimension, analysis_fps
    detector_config = dict(config)
    detector_config["max_dimension"] = config.get("video_max_dimension", config["max_dimension"])
    detector, tracker, writer = Detector(detector_config), None, None
    
    previous_render = None
    paused = False
    frame_index = 0
    last_clear_time = None
    last_preview_clock = 0.0
    try:
        while True:
            preview_message = None
            if stop and stop():
                break
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            if tracker is None:
                tracker = Tracker(config, frame.shape)
            
            if frame_index % stride == 0:
                
                #analisa o quadro e retorna os objetos encontrados.
                detections = detector.detect(frame)
                
                #detecta se alguma mão apareceu, se aparecer, o programa ativa uma pausa. Quando desaparece, inicia uma espera de 0,20 segundos,
                #depois do intervalo, volta normalmente.
                hand = hand_present(frame)
                if hand:
                    last_clear_time = None
                elif last_clear_time is None:
                    last_clear_time = timestamp
                paused = hand or (
                    last_clear_time is not None
                    and timestamp - last_clear_time < .20
                    )
                
                #atualiza o tracker.
                tracker.update(detections, timestamp, paused)
                
                #calcula o progresso e quantos itens estão sendo encontrados.
                if progress and time.perf_counter() - last_preview_clock >= .10:
                    percent = f"{100 * frame_index / max(1, frame_count):.0f}%"
                    preview_message = f"{label}: {percent} | {len(tracker.active())} objetos"
                    
            #retorna os objetos que o rastreador considera ativos.
            active = tracker.active()
            
            #recebe o quadro original e as informações dos objetos para gerar uma imagem anotada.
            previous_render = annotate(frame, active, tracker, timestamp, paused)
            if writer is None:
                writer = cv2.VideoWriter(str(dest / "resultado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         source_fps, (previous_render.shape[1], previous_render.shape[0]))
                if not writer.isOpened():
                    raise OSError("O codificador MP4 não pôde ser iniciado")
            writer.write(previous_render)
            frame_index += 1
            if preview_message:
                progress(preview_message, previous_render)
                last_preview_clock = time.perf_counter()
    finally:
        cap.release()
        if writer:
            writer.release()
    if tracker is None or previous_render is None:
        raise ValueError("O vídeo não contém quadros decodificáveis")
    write_image(dest / "ultimo_quadro.jpg", previous_render)
    if progress:
        progress(f"Concluído: {label}", previous_render)
    return dest / "resultado.mp4"
