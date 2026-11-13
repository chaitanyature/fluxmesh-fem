"""Portable output and geometry checks that work from the published arrays."""

import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from .input import problem_from
from .mesh import Mesh


def write_vtk(path, mesh, values, flux, indicator):
    with path.open("w", encoding="ascii") as f:
        f.write(
            "# vtk DataFile Version 3.0\nFluxmesh adaptive diffusion\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        )
        f.write(f"POINTS {len(values)} double\n")
        np.savetxt(
            f, np.column_stack((mesh.points, np.zeros(len(values)))), fmt="%.16g"
        )
        f.write(f"CELLS {len(mesh.cells)} {4 * len(mesh.cells)}\n")
        np.savetxt(
            f, np.column_stack((np.full(len(mesh.cells), 3), mesh.cells)), fmt="%d"
        )
        f.write(f"CELL_TYPES {len(mesh.cells)}\n")
        np.savetxt(f, np.full(len(mesh.cells), 5), fmt="%d")
        f.write(f"POINT_DATA {len(values)}\nSCALARS u double 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, values, fmt="%.16g")
        f.write(f"CELL_DATA {len(mesh.cells)}\nVECTORS flux double\n")
        np.savetxt(f, np.column_stack((flux, np.zeros(len(flux)))), fmt="%.16g")
        f.write("SCALARS indicator double 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, indicator, fmt="%.16g")


def bundle(dest, solution, history, settings):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"choose a new output directory: {dest}")
    mesh = solution.mesh
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".fluxmesh-") as staging:
        output = Path(staging) / "run"
        output.mkdir()
        (output / "run.json").write_text(
            json.dumps(
                {"format": 1, "settings": settings, "metrics": solution.metrics},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        np.savez_compressed(
            output / "field.npz",
            points=mesh.points,
            triangles=mesh.cells,
            values=solution.values,
            flux=solution.flux,
            indicator=solution.indicator,
        )
        write_vtk(
            output / "field.vtk",
            mesh,
            solution.values,
            solution.flux,
            solution.indicator,
        )
        with (output / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        output.rename(dest)


def audit(dest, tol=1e-8):
    """Reassemble the weak residual by direct integration without calling the sparse solver."""
    from .solver import Q, W

    dest = Path(dest)
    info = json.loads((dest / "run.json").read_text())
    if info["format"] != 1:
        raise ValueError("unknown run format")
    problem = problem_from(info["settings"]["case"], info["settings"].get("initial", 4))
    with np.load(dest / "field.npz", allow_pickle=False) as arrays:
        mesh = Mesh(arrays["points"], arrays["triangles"])
        val, flux, eta = arrays["values"], arrays["flux"], arrays["indicator"]
    T = mesh.cells
    xy = mesh.points[T]
    if (
        val.shape != (len(mesh.points),)
        or flux.shape != (len(T), 2)
        or eta.shape != (len(T),)
    ):
        raise ValueError("field dimensions disagree")
    if not all(np.isfinite(x).all() for x in [val, flux, eta]) or np.any(eta < 0):
        raise ValueError("invalid field values")
    free = np.setdiff1d(np.arange(len(val)), mesh.boundary)
    # Fit an affine polynomial to each local triangle's three nodal values.
    mat = np.concatenate((np.ones((*T.shape, 1)), xy), axis=2)
    coeff = np.linalg.solve(mat, val[T][..., None])[:, :, 0]
    grad = coeff[:, 1:]
    basis = np.linalg.inv(mat)[:, 1:, :]
    k = problem.conductivity(xy.mean(axis=1))
    xyq = np.einsum("qi,tid->tqd", Q, xy)
    f = problem.source(xyq)
    predicted = -k[:, None] * grad
    rhs = np.einsum("qi,tq,q,t->ti", Q, f, W, mesh.area)
    local = mesh.area[:, None] * k[:, None] * np.einsum("td,tdi->ti", grad, basis) - rhs
    res = np.zeros(len(val))
    np.add.at(res, T.ravel(), local.ravel())
    expected_area = 3.0 if problem.domain == "corner" else 1.0
    with (dest / "history.csv").open() as history:
        steps = list(csv.DictReader(history))
    geometric_boundary_error = 0.0
    for a, b in mesh.points[mesh.topology.edges[mesh.topology.counts == 1]]:
        candidates = (
            [(0, 0), (0, 1), (1, 0), (1, 1)]
            if problem.domain == "square"
            else [(0, -1), (0, 1), (1, -1), (1, 1)]
        )
        if problem.domain == "corner":
            if max(a[1], b[1]) <= 0:
                candidates.append((0, 0))
            if min(a[0], b[0]) >= 0:
                candidates.append((1, 0))
        geometric_boundary_error = max(
            geometric_boundary_error,
            min(max(abs(a[d] - v), abs(b[d] - v)) for d, v in candidates),
        )
    fields = {
        "free_residual": float(np.max(abs(res[free]), initial=0)),
        "boundary_values": float(
            np.max(
                np.abs(
                    val[mesh.boundary] - problem.boundary(mesh.points[mesh.boundary])
                )
            )
        ),
        "flux_disagreement": float(np.max(np.abs(predicted - flux))),
        "area_error": abs(float(mesh.area.sum()) - expected_area),
        "geometric_boundary_error": geometric_boundary_error,
        "estimator_disagreement": abs(
            float(np.linalg.norm(eta)) - info["metrics"]["estimator"]
        ),
    }
    if problem.gradient is not None:
        energy = float(
            np.sqrt(
                np.sum(
                    mesh.area
                    * k
                    * (
                        np.sum((grad[:, None, :] - problem.gradient(xyq)) ** 2, axis=2)
                        @ W
                    )
                )
            )
        )
        fields["energy_disagreement"] = abs(energy - info["metrics"]["energy_error"])
    counts_ok = (
        info["metrics"]["triangles"] == len(T)
        and info["metrics"]["vertices"] == len(val)
        and bool(steps)
        and int(steps[-1]["triangles"]) == len(T)
    )
    return {
        "valid": counts_ok
        and all(np.isfinite(x) and x <= tol for x in fields.values()),
        "checks": fields,
    }
