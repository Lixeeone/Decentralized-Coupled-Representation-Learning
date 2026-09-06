"""Tangent projections and retractions for the spatial Euler step.

Sphere and torus are compact boundaryless examples. Strips and cylinders have
boundaries; their clipping behavior is a numerical stress protocol, not a new
boundaryless-manifold theorem. The empirical ball uses ambient tangent vectors.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Domain:
    kind: str = "ball"
    radius: float = 5.0
    center: np.ndarray | None = None
    major_radius: float = 2.0
    minor_radius: float = 0.7

    def __post_init__(self):
        if self.kind not in {"ball", "sphere", "torus", "swiss_roll", "s_curve", "cylinder", "hyperboloid", "identity"}:
            raise ValueError(f"unsupported domain: {self.kind}")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")
        if not (np.isfinite(self.major_radius) and np.isfinite(self.minor_radius)
                and 0 < self.minor_radius < self.major_radius):
            raise ValueError("torus radii must satisfy 0 < minor < major")
        if self.center is not None:
            self.center = np.asarray(self.center, dtype=float)

    def _normal(self, x):
        if self.kind == "sphere":
            return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-15)
        if self.kind == "torus":
            rho = np.maximum(np.linalg.norm(x[:, :2], axis=1), 1e-15)
            c = np.column_stack([self.major_radius * x[:, 0] / rho,
                                 self.major_radius * x[:, 1] / rho, np.zeros(len(x))])
            v = x - c
        elif self.kind in {"swiss_roll", "s_curve"}:
            t = self._curve_parameter(x)
            _, d, _ = self._curve(t)
            v = np.column_stack([-d[:, 1], np.zeros(len(x)), d[:, 0]])
        elif self.kind == "cylinder":
            v = np.column_stack([x[:, 0], x[:, 1], np.zeros(len(x))])
        elif self.kind == "hyperboloid":
            v = x * np.array([1., 1., -1.])
        else:
            return np.zeros_like(x)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-15)

    def tangent(self, x, v):
        if self.kind in {"identity", "ball"}:
            return v
        normal = self._normal(x)
        return v - np.sum(v * normal, axis=1, keepdims=True) * normal

    def _curve(self, t):
        if self.kind == "swiss_roll":
            p = np.stack([t * np.cos(t), t * np.sin(t)], axis=-1)
            d = np.stack([np.cos(t) - t * np.sin(t), np.sin(t) + t * np.cos(t)], axis=-1)
            dd = np.stack([-2 * np.sin(t) - t * np.cos(t), 2 * np.cos(t) - t * np.sin(t)], axis=-1)
        else:
            s = np.where(t < 0, -1., 1.)
            p = np.stack([np.sin(t), s * (np.cos(t) - 1)], axis=-1)
            d = np.stack([np.cos(t), -s * np.sin(t)], axis=-1)
            dd = np.stack([-np.sin(t), -s * np.cos(t)], axis=-1)
        return p, d, dd

    def _curve_parameter(self, x):
        # A dense coarse search followed by local Newton refinement. Candidates
        # include both interval endpoints; choose the lowest distance attained.
        lo, hi = (1.5 * np.pi, 4.5 * np.pi) if self.kind == "swiss_roll" else (-1.5 * np.pi, 1.5 * np.pi)
        grid = np.linspace(lo, hi, 96)
        p, _, _ = self._curve(grid)
        xz = x[:, [0, 2]]
        dist = np.sum((xz[:, None, :] - p[None, :, :]) ** 2, axis=2)
        index = np.argmin(dist, axis=1)
        t = grid[index]
        best_t, best_dist = t.copy(), dist[np.arange(len(x)), index]
        for _ in range(8):
            p, d, dd = self._curve(t)
            num = np.sum((p - xz) * d, axis=1)
            den = np.sum(d * d + (p - xz) * dd, axis=1)
            t = np.clip(t - np.clip(num / np.maximum(den, 1e-8), -0.2, 0.2), lo, hi)
            p, _, _ = self._curve(t)
            cost = np.sum((p - xz) ** 2, axis=1)
            better = cost < best_dist
            best_t[better], best_dist[better] = t[better], cost[better]
        return best_t

    def retract(self, x):
        x = np.asarray(x, dtype=float)
        if self.kind == "identity":
            return x.copy()
        if self.kind == "ball":
            center = 0.0 if self.center is None else self.center
            v = x - center
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            return center + v * np.minimum(1, self.radius / np.maximum(norms, 1e-15))
        if x.shape[1] != 3:
            raise ValueError(f"{self.kind} requires ambient dimension 3")
        if self.kind == "sphere":
            norm = np.linalg.norm(x, axis=1, keepdims=True)
            out = self.radius * x / np.maximum(norm, 1e-15)
            out[norm[:, 0] < 1e-15] = [self.radius, 0, 0]
            return out
        if self.kind == "torus":
            phi = np.arctan2(x[:, 1], x[:, 0])
            theta = np.arctan2(x[:, 2], np.linalg.norm(x[:, :2], axis=1) - self.major_radius)
            rho = self.major_radius + self.minor_radius * np.cos(theta)
            return np.column_stack([rho * np.cos(phi), rho * np.sin(phi), self.minor_radius * np.sin(theta)])
        if self.kind in {"swiss_roll", "s_curve"}:
            p, _, _ = self._curve(self._curve_parameter(x))
            height = 6.0 if self.kind == "swiss_roll" else 1.0
            return np.column_stack([p[:, 0], np.clip(x[:, 1], -height, height), p[:, 1]])
        phi = np.arctan2(x[:, 1], x[:, 0])
        z = np.clip(x[:, 2], -1, 1)
        if self.kind == "hyperboloid":
            rho0 = np.linalg.norm(x[:, :2], axis=1)
            for _ in range(12):
                r = np.sqrt(1 + z * z)
                derivative = (r - rho0) * z / r + z - x[:, 2]
                second = z * z / (r * r) + (r - rho0) / r ** 3 + 1
                z = np.clip(z - derivative / np.maximum(second, 0.1), -1, 1)
            rho = np.sqrt(1 + z * z)
        else:
            rho = np.ones(len(x)) * self.radius
        return np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])


def make_domain(spec, x):
    spec = dict(spec)
    kind = spec.get("kind", "ball")
    if kind == "ball":
        center = x.mean(axis=0)
        spec.setdefault("center", center)
        spec.setdefault("radius", max(float(np.linalg.norm(x - center, axis=1).max()), 1e-8) * (1 + 1e-12))
    return Domain(**spec)
