"""Dados das detecções e medidas em pixels, incluindo comprimentos curvos."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import cv2
import numpy as np

from .hsv import color_distribution
from .mask_utils import contours

LABELS = ("fita_rosa", "minhoca", "ferramenta")


# Estrutura dos dados que uma detecção entrega ao rastreador e à renderização.
# kind = classe; bbox = caixa (x,y,largura,altura); center = centro;
# polygon = quatro cantos da caixa inclinada; area_px = área;
# minor_px = largura; length_px = comprimento; colors = frações de cores;
# object_id = identificador usado no resultado.
@dataclass
class Detection:
    kind: str
    bbox: tuple
    center: tuple
    polygon: list
    area_px: float
    minor_px: float
    length_px: float
    colors: dict
    object_id: int = 0


# Reduz uma região grossa a linhas com cerca de um pixel de largura
# (esqueleto) pelo algoritmo de Zhang-Suen.
def thin(mask):
    """Afinamento Zhang-Suen, sem opencv-contrib."""
    # Transforma a máscara em 0/1 e adiciona borda para examinar vizinhos.
    pixels = np.pad(mask > 0, 1).astype(np.uint8)
    # Repete até estabilizar ou atingir o limite de 160 iterações.
    for _ in range(160):
        changed = False
        # Zhang-Suen faz duas subetapas com regras diferentes de remoção.
        for step in (0, 1):
            p = pixels[1:-1, 1:-1]
            n = [pixels[:-2, 1:-1], pixels[:-2, 2:], pixels[1:-1, 2:],
                 pixels[2:, 2:], pixels[2:, 1:-1], pixels[2:, :-2],
                 pixels[1:-1, :-2], pixels[:-2, :-2]]
            # Conta vizinhos ligados e transições 0->1 em torno de cada pixel.
            total = sum(n)
            transitions = sum((n[i] == 0) & (n[(i + 1) % 8] == 1) for i in range(8))
            a, b = ((n[0] * n[2] * n[4], n[2] * n[4] * n[6]) if step == 0
                    else (n[0] * n[2] * n[6], n[0] * n[4] * n[6]))
            # Remove pixels das bordas sem romper a conectividade principal.
            remove = (p == 1) & (total >= 2) & (total <= 6) & (transitions == 1) & (a == 0) & (b == 0)
            if remove.any():
                p[remove] = 0
                changed = True
        if not changed:
            break
    return pixels[1:-1, 1:-1]


# Estima o comprimento do objeto acompanhando seu esqueleto,
# mais adequado a minhocas/fitas curvas do que uma reta ponta-a-ponta.
# O resultado está em pixels da imagem utilizada na análise.
def centerline_length(mask):
    # Obtém as posições de todos os pixels pertencentes ao objeto.
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = np.pad(mask[y0:y1, x0:x1], 2)
    # Afina a máscara e guarda coordenadas dos pixels do esqueleto.
    points = set(map(tuple, np.argwhere(thin(crop))))
    if len(points) < 2:
        return float(max(x1 - x0, y1 - y0))
    # Vizinho horizontal/vertical custa 1; diagonal custa raiz de 2.
    neighbors = [(dy, dx, math.hypot(dx, dy)) for dy in (-1, 0, 1)
                 for dx in (-1, 0, 1) if dx or dy]

    # Busca, na componente do esqueleto, o ponto com maior distância
    # acumulada a partir do ponto inicial (fila de prioridades).
    def farthest(start):
        distances, queue = {start: 0.0}, [(0.0, start)]
        while queue:
            distance, point = heapq.heappop(queue)
            if distance > distances[point]:
                continue
            for dy, dx, cost in neighbors:
                nxt = (point[0] + dy, point[1] + dx)
                new_distance = distance + cost
                if nxt in points and new_distance < distances.get(nxt, float("inf")):
                    distances[nxt] = new_distance
                    heapq.heappush(queue, (new_distance, nxt))
        end = max(distances, key=distances.get)
        return end, distances[end], set(distances)

    remaining = points.copy()
    # A transformada de distância estima quão longe cada pixel está
    # do fundo; os valores nas pontas ajudam a corrigir o comprimento.
    radii = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    length = 0.0
    # Se o esqueleto tiver pedaços desconectados, avalia cada componente
    # e conserva a maior estimativa de comprimento.
    while remaining:
        start, _, component = farthest(next(iter(remaining)))
        end, distance, _ = farthest(start)
        length = max(length, distance + float(radii[start] + radii[end]))
        remaining -= component
    return length


class Measurer:
    """Mede as máscaras e reaproveita medidas de objetos estáveis nos vídeos."""

    def __init__(self):
        self.measure_cache = []

    # Para cada máscara aceita: calcula geometria, cores e IDs e devolve
    # uma lista de objetos Detection para render.py/tracking.py.
    def measure(self, objects, hsv, scale, width):
        detections = []
        for kind, mask in objects:
            cs = contours(mask)
            if not cs:
                continue
            contour = max(cs, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            # Solidity = área real / área da envoltória convexa;
            # ajuda a reconhecer a forma para reutilizar medidas em cache.
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            rect = cv2.minAreaRect(contour)
            short, long = sorted(rect[1])
            solidity = area / max(1, hull_area)
            length = long
            # Nos doces, mede o comprimento da linha central, que pode ser curva.
            # Nas ferramentas, usa o eixo maior do retângulo inclinado.
            if kind != "ferramenta":
                cached = next((e for e in reversed(self.measure_cache)
                               if e[0] == kind and math.dist(e[1], rect[0]) < width * .025
                               and abs(e[2] - solidity) < .06 and .90 < long / e[3] < 1.10), None)
                # Se houver medição equivalente recente, reutiliza o fator curva/eixo.
                # Caso contrário, calcula o esqueleto e salva o resultado no cache.
                if cached:
                    length = long * cached[4]
                else:
                    length = max(centerline_length(mask), long)
                    self.measure_cache.append((kind, rect[0], solidity, long, length / max(long, 1)))
                    self.measure_cache = self.measure_cache[-80:]
            x, y, bw, bh = cv2.boundingRect(contour)
            # Divide por scale para reportar posições e medidas na imagem original.
            detections.append(Detection(
                kind=kind,
                bbox=tuple(round(v / scale, 2) for v in (x, y, bw, bh)),
                center=tuple(round(v / scale, 2) for v in rect[0]),
                polygon=np.round(cv2.boxPoints(rect) / scale, 2).tolist(),
                area_px=round(area / scale ** 2, 2), minor_px=round(short / scale, 2),
                length_px=round(length / scale, 2), colors=color_distribution(hsv, mask, kind)
            ))
        # Ordena primeiro de cima para baixo e depois da esquerda para direita.
        detections.sort(key=lambda d: (d.center[1], d.center[0]))
        for index, detection in enumerate(detections, 1):
            detection.object_id = index
        return detections
