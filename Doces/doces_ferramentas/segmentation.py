"""Coordena as máscaras de cor, os filtros de formato e a separação dos objetos."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .hsv import prepare_hsv, pink_mask, tool_mask, gummy_mask, metal_mask, gummy_color_valid
from .watershed import split_touching
from .mask_utils import kernel, contours, filled, contour_mask


class Segmenter:
    """Executa HSV → fitas → ferramentas → minhocas e devolve máscaras e HSV."""

    def __init__(self, config):
        self.config = config

    def segment(self, image):
        h, w = image.shape[:2]
        hsv, metal_highlight, gain = prepare_hsv(image)
        # occupied guarda os pixels já atribuídos a fitas/ferramentas.
        # Isso evita classificá-los novamente como minhocas coloridas.
        objects, occupied = [], np.zeros((h, w), np.uint8)
        # Detecta PRIMEIRO as fitas rosas.
        for kind, mask in self._pink(hsv, (h, w)):
            objects.append((kind, mask))
            occupied |= cv2.dilate(mask, kernel(5))
        # Depois procura ferramentas de cabo amarelo e suas hastes.
        for kind, mask in self._tools(hsv, metal_highlight, gain):
            objects.append((kind, mask))
            occupied |= cv2.dilate(mask, kernel(5))
        # Por último procura minhocas, sem reutilizar as regiões ocupadas.
        objects.extend(self._gummies(hsv, occupied))
        return objects, hsv

    # FITAS ROSAS: máscara HSV + fechamento + filtro de tamanho/formato.
    # Aceita somente regiões suficientemente alongadas da máscara rosa.
    def _pink(self, hsv, shape):
        h, w = shape
        pink = pink_mask(hsv, self.config)
        for contour in contours(pink):
            short, long = sorted(cv2.minAreaRect(contour)[1])
            if (cv2.contourArea(contour) <= h * w * self.config["min_area_fraction"]
                    or long <= h * .15 or long / max(short, 1) < 3.5):
                continue
            yield "fita_rosa", contour_mask(pink.shape, contour)

    # FERRAMENTAS: acha cabo amarelo e procura haste metálica próxima,
    # orientada na mesma direção do cabo.
    def _tools(self, hsv, metal_highlight, gain):
        h, w = hsv.shape[:2]
        yellow = tool_mask(hsv, self.config, gain)
        for contour in contours(yellow):
            short, long = sorted(cv2.minAreaRect(contour)[1])
            if (cv2.contourArea(contour) < h * w * self.config["tool_min_area_fraction"]
                    or short < w * self.config["tool_min_width_fraction"]):
                continue
            # Convex hull envolve as partes do cabo com seu contorno convexo.
            hull = cv2.convexHull(contour)
            mask = contour_mask(yellow.shape, hull)
            box = cv2.boxPoints(cv2.minAreaRect(hull))
            edge1, edge2 = box[1] - box[0], box[2] - box[1]
            axis = edge1 if np.linalg.norm(edge1) > np.linalg.norm(edge2) else edge2
            # Calcula a direção principal (eixo longo) da ferramenta.
            axis /= np.linalg.norm(axis)
            perpendicular = np.array([-axis[1], axis[0]])
            center = np.array(cv2.minAreaRect(hull)[0])
            yy, xx = np.indices((h, w))
            dx, dy = xx - center[0], yy - center[1]
            along, across = dx * axis[0] + dy * axis[1], dx * perpendicular[0] + dy * perpendicular[1]
            # Procura metal somente em uma faixa orientada junto ao cabo,
            # reduzindo falsos positivos pelo fundo da imagem.
            corridor = (np.abs(along) < long * 1.25) & (np.abs(across) < short * .80)
            # Mede distância de regiões candidatas ao cabo, para preferir
            # segmentos metálicos próximos.
            distance_to_body = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
            shaft_candidates = []
            # Primeira passagem rigorosa; segunda, mais permissiva se necessário.
            for contrast, saturation in ((18, 90), (8, 120)):
                metal = metal_mask(hsv, metal_highlight, corridor, contrast, saturation)
                metal[cv2.dilate(mask, kernel(3)) > 0] = 0
                for piece in contours(metal):
                    metal_short, metal_long = sorted(cv2.minAreaRect(piece)[1])
                    if (metal_long < long * .35 or metal_long / max(metal_short, 1) < 2.5
                            or metal_short > short * .45):
                        continue
                    box = cv2.boxPoints(cv2.minAreaRect(piece))
                    edges = (box[1] - box[0], box[2] - box[1])
                    direction = max(edges, key=np.linalg.norm)
                    # Produto escalar: a haste precisa estar quase alinhada ao cabo.
                    if abs(np.dot(direction / max(metal_long, 1), axis)) < .85:
                        continue
                    points = piece[:, 0, :]
                    distances = distance_to_body[points[:, 1], points[:, 0]]
                    index = int(np.argmin(distances))
                    if distances[index] < short * .85:
                        shaft_candidates.append((float(distances[index]), -metal_long,
                                                 piece, points[index], metal_short))
                if shaft_candidates:
                    break
            if shaft_candidates:
                _, _, shaft, near_shaft, width = min(shaft_candidates, key=lambda item: (item[0], item[1]))
                body_points = np.concatenate(contours(mask))[:, 0, :]
                near_body = body_points[np.argmin(np.sum((body_points - near_shaft) ** 2, axis=1))]
                cv2.drawContours(mask, [shaft], -1, 255, -1)
                cv2.line(mask, tuple(near_body), tuple(near_shaft), 255, max(2, round(width)))
                mask = filled(mask)
            # yield devolve cada ferramenta identificada sem montar lista intermediária.
            yield "ferramenta", mask

    # MINHOCAS: combina várias faixas de cor e valida o formato
    # antes de aplicar o Watershed quando houver possível contato.
    def _gummies(self, hsv, occupied):
        h, w = hsv.shape[:2]
        gummy = gummy_mask(hsv, self.config)
        # Exclui locais já atribuídos a fitas ou ferramentas.
        gummy[occupied > 0] = 0
        # Fechamento conecta pequenas interrupções na região colorida.
        gummy = cv2.morphologyEx(gummy, cv2.MORPH_CLOSE, kernel(w / 34))
        # Abertura = erosão seguida de dilatação; remove ruídos isolados.
        gummy = cv2.morphologyEx(gummy, cv2.MORPH_OPEN, kernel(3))
        gummy[occupied > 0] = 0
        for contour in contours(gummy):
            area = cv2.contourArea(contour)
            x, y, bw, bh = cv2.boundingRect(contour)
            if (area < h * w * self.config["min_area_fraction"] or area > h * w * .035
                    or x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
                continue
            mask = contour_mask(gummy.shape, contour)
            # Aceita regiões de aspecto tubular, evitando confundir objetos largos
            # com minhocas. Se duas estiverem coladas, tenta separá-las.
            if self._is_gummy(mask, hsv, h * w):
                for part in split_touching(mask, hsv, h * w, self.config["min_area_fraction"]):
                    yield "minhoca", part

    # Filtro de forma: verifica espessura e proporção entre área e diâmetro.
    # Também rejeita regiões com muitas lacunas escuras ou pouca cor quente.
    def _is_gummy(self, mask, hsv, image_area):
        x, y, w, h = cv2.boundingRect(mask)
        crop = np.pad(mask[y:y+h, x:x+w], 1)
        # Diâmetro estimado = duas vezes a maior distância até o fundo.
        diameter = 2 * float(cv2.distanceTransform(crop, cv2.DIST_L2, 5).max())
        area = cv2.contourArea(max(contours(mask), key=cv2.contourArea))
        if diameter > .06 * math.sqrt(image_area) or area < 2 * diameter ** 2:
            return False
        return gummy_color_valid(hsv, mask)
