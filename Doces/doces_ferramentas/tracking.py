"""Mantém os IDs e contabiliza entradas e saídas nos vídeos.

Compara cada detecção com os objetos anteriores por classe, posição, área e
cores. Confirma mudanças após persistência temporal e pausa a atualização
durante oclusões. Guarda os contadores e o último evento para o painel visual.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .vision import Detection, LABELS


@dataclass
class Track:
    id: int
    detection: Detection
    confirmed: bool = False
    seen_seconds: float = 0.0
    missing_seconds: float = 0.0


class Tracker:
    """Estado dos objetos acompanhados ao longo dos quadros já observados."""

    def __init__(self, config, frame_shape):
        self.config = config
        self.diagonal = math.hypot(*frame_shape[:2])
        self.tracks = {}
        self.next_id = 1
        self.last_event = None
        self.entered = {kind: 0 for kind in LABELS}
        self.exited = {kind: 0 for kind in LABELS}
        self.initialized = False
        self.previous_time = None

    def _new(self, det, confirmed=False):
        id_ = self.next_id
        self.next_id += 1
        det.object_id = id_
        track = Track(id_, det, confirmed=confirmed)
        self.tracks[id_] = track

    def _event(self, action, track, timestamp):
        det = track.detection
        counter = self.entered if action == "entrada" else self.exited
        counter[det.kind] += 1
        self.last_event = {"tempo_s": round(timestamp, 3), "acao": action, "id": track.id}

    def update(self, detections, timestamp, occluded=False):
        dt = max(0, timestamp - self.previous_time) if self.previous_time is not None else 0
        self.previous_time = timestamp
        if not self.initialized:
            # O inventário do primeiro quadro é a linha de base, não uma entrada.
            for det in detections:
                self._new(det, confirmed=True)
            self.initialized = True
            return
        if occluded:
            # Pausa inclusive os cronômetros de ausência enquanto a mão oclui a mesa.
            return

        # Associação por distância e tamanho, dentro da mesma classe. Ordenar os
        # custos evita que a ordem dos contornos determine a identidade atribuída.
        pairs = []
        for id_, track in self.tracks.items():
            old = track.detection
            for j, det in enumerate(detections):
                if det.kind != old.kind:
                    continue
                distance = math.dist(old.center, det.center)
                area_ratio = det.area_px / max(old.area_px, 1)
                if distance > self.diagonal * self.config["match_distance_fraction"] or not .30 < area_ratio < 3.3:
                    continue
                color_difference = 1 - sum(min(old.colors.get(k, 0), det.colors.get(k, 0)) for k in old.colors)
                cost = distance / self.diagonal + .035 * abs(math.log(area_ratio)) + .035 * color_difference
                pairs.append((cost, id_, j))
        used_tracks, used_detections = set(), set()
        for _, id_, j in sorted(pairs):
            if id_ in used_tracks or j in used_detections:
                continue
            track = self.tracks[id_]
            det = detections[j]
            det.object_id = id_
            track.detection = det
            track.seen_seconds += dt
            track.missing_seconds = 0
            used_tracks.add(id_)
            used_detections.add(j)
            if not track.confirmed and track.seen_seconds >= self.config["confirmation_seconds"]:
                track.confirmed = True
                self._event("entrada", track, timestamp)
        for id_ in list(self.tracks):
            if id_ in used_tracks:
                continue
            track = self.tracks[id_]
            track.missing_seconds += dt
            # Uma detecção intermitente não acumula confirmações descontínuas.
            if not track.confirmed:
                track.seen_seconds = 0
            if track.missing_seconds >= self.config["removal_seconds"]:
                if track.confirmed:
                    self._event("saida", track, timestamp)
                del self.tracks[id_]
        for j, det in enumerate(detections):
            if j not in used_detections:
                self._new(det)

    def active(self):
        return [track.detection for track in self.tracks.values() if track.confirmed]


