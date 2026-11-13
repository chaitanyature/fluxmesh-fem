"""Declarative smooth heat sources with mesh-aligned material interfaces."""

import math
import numpy as np
from .cases import CASES, Problem


def numbers(values, length):
    if (
        not isinstance(values, list)
        or len(values) != length
        or not all(type(v) in (int, float) and math.isfinite(v) for v in values)
    ):
        raise ValueError(f"expected {length} finite numbers")
    return values


def problem_from(spec, initial):
    if isinstance(spec, str):
        if spec not in CASES:
            raise ValueError(f"unknown case {spec}")
        return CASES[spec]
    if not isinstance(spec, dict) or set(spec) != {
        "domain",
        "material",
        "boundary",
        "sources",
    }:
        raise ValueError("expected domain, material, boundary, and sources")
    domain = spec["domain"]
    if domain not in ("corner", "square"):
        raise ValueError("invalid domain")
    material = spec["material"]
    if not isinstance(material, dict) or set(material) != {
        "x_breaks",
        "conductivities",
    }:
        raise ValueError("invalid material map")
    breaks = material["x_breaks"]
    if not isinstance(breaks, list):
        raise ValueError("x_breaks must be an array")
    numbers(breaks, len(breaks))
    coeff = numbers(material["conductivities"], len(breaks) + 1)
    if any(k <= 0 for k in coeff):
        raise ValueError("conductivities must be positive")
    xmin = -1 if domain == "corner" else 0
    if sorted(set(breaks)) != breaks or any(not xmin < x < 1 for x in breaks):
        raise ValueError("material breaks must increase inside the domain")
    for x in breaks:
        i = (x - xmin) * initial / (1 - xmin)
        if not math.isclose(i, round(i), abs_tol=1e-12, rel_tol=0):
            raise ValueError("material break must align with the initial grid")
    c, dx, dy = numbers(spec["boundary"], 3)

    def boundary(x):
        return c + dx * x[..., 0] + dy * x[..., 1]

    def kappa(x):
        return np.asarray(coeff)[np.searchsorted(breaks, x[..., 0], side="right")]

    sources = spec["sources"]
    if not isinstance(sources, list):
        raise ValueError("sources must be an array")
    spots = []
    for spot in sources:
        if not isinstance(spot, dict) or set(spot) != {"center", "sigma", "amplitude"}:
            raise ValueError("invalid Gaussian source")
        a, b = numbers(spot["center"], 2)
        sigma, amplitude = numbers([spot["sigma"], spot["amplitude"]], 2)
        if sigma <= 0:
            raise ValueError("source sigma must be positive")
        spots.append((a, b, sigma, amplitude))

    def forcing(x):
        out = np.zeros(x.shape[:-1])
        for a, b, sigma, amplitude in spots:
            out += amplitude * np.exp(
                -0.5 * (((x[..., 0] - a) / sigma) ** 2 + ((x[..., 1] - b) / sigma) ** 2)
            )
        return out

    return Problem("custom", domain, kappa, forcing, boundary, None)
