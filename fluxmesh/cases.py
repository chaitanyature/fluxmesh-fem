from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    domain: str
    conductivity: object
    source: object
    exact: object
    gradient: object


def one(x):
    return np.ones(x.shape[:-1])


def zero(x):
    return np.zeros(x.shape[:-1])


def smooth(x):
    return np.sin(np.pi * x[..., 0]) * np.sin(np.pi * x[..., 1])


def smooth_grad(x):
    a, b = x[..., 0] * np.pi, x[..., 1] * np.pi
    return np.pi * np.stack((np.cos(a) * np.sin(b), np.sin(a) * np.cos(b)), axis=-1)


def polar(x):
    a, b = x[..., 0], x[..., 1]
    return np.hypot(a, b), np.mod(np.arctan2(b, a), 2 * np.pi)


def corner(x):
    r, theta = polar(x)
    return r ** (2 / 3) * np.sin(2 / 3 * theta)


def corner_grad(x):
    r, theta = polar(x)
    factor = (2 / 3) * np.maximum(r, 1e-100) ** (-1 / 3)
    return factor[..., None] * np.stack(
        (np.sin(-theta / 3), np.cos(-theta / 3)), axis=-1
    )


def conductivity(x):
    return np.where(x[..., 0] < 0.5, 1.0, 10.0)


def transmission(x):
    a = x[..., 0]
    return np.where(a < 0.5, a, 0.5 + (a - 0.5) / 10)


def transmission_grad(x):
    return np.stack((1 / conductivity(x), zero(x)), axis=-1)


CASES = {
    "smooth": Problem(
        "smooth", "square", one, lambda x: 2 * np.pi**2 * smooth(x), smooth, smooth_grad
    ),
    "corner": Problem("corner", "corner", one, zero, corner, corner_grad),
    "interface": Problem(
        "interface", "square", conductivity, zero, transmission, transmission_grad
    ),
}
