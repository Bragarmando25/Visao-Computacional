"""Desenha as identificações e o painel de resultados sobre a imagem.

Mostra caixas, IDs, classes, cores e medidas em pixels. Nos quadros de vídeo,
acrescenta os contadores de entradas e saídas e o último evento confirmado.
"""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np
from .vision import LABELS

COLORS = {"fita_rosa": (188, 127, 255), "minhoca": (113, 213, 91), "ferramenta": (72, 205, 255)}
NAMES = {"fita_rosa": "Fita rosa", "minhoca": "Minhoca", "ferramenta": "Ferramenta"}


def text(im, value, xy, size=.49, color=(226, 231, 241), thickness=1):
    # A fonte Hershey do OpenCV exige normalizar os acentos para desenhar o texto.
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    cv2.putText(im, value, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def annotate(frame, detections, tracker=None, timestamp=None, paused=False):
    """Devolve a imagem com os objetos marcados e seus resultados escritos."""
    h0, w0 = frame.shape[:2]
    scale = min(1.0, 960 / max(h0, w0))
    display = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
    h, w = display.shape[:2]
    # Dimensão fixa durante todo o vídeo, incluindo quando a contagem muda.
    height = max(h, 832)
    panel_width = 430
    canvas = np.full((height, w + panel_width, 3), (24, 20, 18), np.uint8)
    canvas[:h, :w] = display
    for det in detections:
        color = COLORS[det.kind]
        polygon = (np.array(det.polygon) * scale).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        x, y = int(det.bbox[0] * scale), int(det.bbox[1] * scale)
        label = f"#{det.object_id}"
        tx, ty = max(0, min(w - 55, x)), max(22, y - 4)
        cv2.rectangle(canvas, (tx, ty - 19), (tx + 55, ty + 3), color, -1)
        text(canvas, label, (tx + 4, ty - 2), .48, (20, 24, 28), 1)
    x = w + 20
    text(canvas, "DOCES + FERRAMENTAS", (x, 32), .66, (245, 245, 250), 2)
    subtitle = f"Video | {timestamp:.2f}s" if timestamp is not None else "Analise de imagem"
    text(canvas, subtitle, (x, 57), .46, (151, 164, 184))
    counts = {kind: sum(d.kind == kind for d in detections) for kind in LABELS}
    text(canvas, f"Total: {len(detections)}   Doces: {counts['fita_rosa'] + counts['minhoca']}   Ferramentas: {counts['ferramenta']}", (x, 86), .48)
    text(canvas, f"Fitas: {counts['fita_rosa']}  |  Minhocas: {counts['minhoca']}", (x, 110), .48)
    if tracker:
        text(canvas, f"Entradas: +{sum(tracker.entered.values())}   Saidas: -{sum(tracker.exited.values())}", (x, 139), .54, (113, 213, 255), 1)
        status = "Intervencao: aguardando mao sair" if paused else "Inventario confirmado"
        text(canvas, status, (x, 163), .43, (103, 193, 255) if paused else (135, 214, 143))
        start_y = 194
    else:
        start_y = 146
    available = height - start_y - 100
    row_height = min(46, max(29, available // max(1, len(detections))))
    font = .43 if row_height >= 36 else .37
    for i, d in enumerate(sorted(detections, key=lambda d: d.object_id)):
        y = start_y + i * row_height
        if y + row_height > height - 82:
            text(canvas, f"Mais {len(detections) - i} objetos na imagem", (x, y), .43)
            break
        cv2.circle(canvas, (x + 4, y - 4), 4, COLORS[d.kind], -1)
        colors = "+".join(d.colors)
        colors = colors.replace("cinza/prata", "prata")
        text(canvas, f"#{d.object_id:02d} {NAMES[d.kind]} | {colors}", (x + 15, y), font)
        text(canvas, f"{d.length_px:.0f} px; largura {d.minor_px:.0f} px", (x + 15, y + 17), font - .03, (160, 173, 193))
    if tracker and tracker.last_event:
        last = tracker.last_event
        text(canvas, f"Ultimo: {last['acao']} #{last['id']} em {last['tempo_s']:.1f}s", (x, height - 63), .42, (113, 213, 255))
    text(canvas, "Medidas em pixels da imagem original", (x, height - 40), .40, (155, 166, 186))
    text(canvas, "Curvas: comprimento ao longo do doce", (x, height - 19), .40, (155, 166, 186))
    # mp4v requer dimensões pares em alguns backends.
    if canvas.shape[0] % 2 or canvas.shape[1] % 2:
        canvas = cv2.copyMakeBorder(canvas, 0, canvas.shape[0] % 2, 0, canvas.shape[1] % 2, cv2.BORDER_CONSTANT)
    return canvas
