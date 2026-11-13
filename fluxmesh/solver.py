from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .mesh import Mesh
from .cases import CASES


# Symmetric seven-point triangle quadrature, exact for polynomials through degree five.
Q = np.array(
    [
        [1 / 3, 1 / 3, 1 / 3],
        [0.05971587178977, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.05971587178977, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.05971587178977],
        [0.797426985353087, 0.1012865073234565, 0.1012865073234565],
        [0.1012865073234565, 0.797426985353087, 0.1012865073234565],
        [0.1012865073234565, 0.1012865073234565, 0.797426985353087],
    ]
)
W = np.array([0.225, *[0.132394152788506] * 3, *[0.125939180544827] * 3])


@dataclass
class Solution:
    mesh: Mesh
    values: np.ndarray
    flux: np.ndarray
    indicator: np.ndarray
    metrics: dict


def solve(mesh, problem):
    cells = mesh.cells
    quadrature = np.einsum("qi,tid->tqd", Q, mesh.points[cells])
    kappa = problem.conductivity(quadrature)
    if not np.isfinite(kappa).all() or np.any(kappa <= 0):
        raise ValueError("conductivity must be positive and finite")
    if not np.allclose(kappa, kappa[:, :1]):
        raise ValueError("material interfaces must follow mesh edges")
    k = kappa[:, 0]
    source = problem.source(quadrature)
    local = np.einsum("tid,tjd,t->tij", mesh.grad, mesh.grad, mesh.area * k)
    rhs = np.einsum("tq,q,qi,t->ti", source, W, Q, mesh.area)
    K = coo_matrix(
        (
            local.ravel(),
            (np.repeat(cells, 3, axis=1).ravel(), np.tile(cells, (1, 3)).ravel()),
        ),
        shape=(len(mesh.points), len(mesh.points)),
    ).tocsr()
    b = np.bincount(cells.ravel(), weights=rhs.ravel(), minlength=len(mesh.points))
    u = np.zeros(len(mesh.points))
    boundary = mesh.boundary
    u[boundary] = problem.exact(mesh.points[boundary])
    free = np.setdiff1d(np.arange(len(u)), boundary)
    if len(free):
        u[free] = spsolve(K[free][:, free], (b - K @ u)[free])
    grad = np.einsum("ti,tid->td", u[cells], mesh.grad)
    flux = -k[:, None] * grad
    t = mesh.topology
    interior = t.counts == 2
    direction = mesh.edge_vectors[interior] / mesh.length[interior, None]
    normal = np.column_stack((direction[:, 1], -direction[:, 0]))
    lhs, rhs_cell = t.owners[interior].T
    jumps = np.einsum("ij,ij->i", flux[lhs] - flux[rhs_cell], normal)
    eta2 = mesh.diameter**2 * mesh.area * ((source**2) @ W)
    edge_int = 0.5 * mesh.length[interior] ** 2 * jumps**2
    np.add.at(eta2, lhs, edge_int)
    np.add.at(eta2, rhs_cell, edge_int)
    values_q = u[cells] @ Q.T
    l2 = np.sum(mesh.area * ((values_q - problem.exact(quadrature)) ** 2 @ W))
    error = grad[:, None, :] - problem.gradient(quadrature)
    energy = np.sum(mesh.area * k * (np.sum(error**2, axis=2) @ W))
    residual = K @ u - b
    # Summing all boundary reactions balances the integrated source for this Dirichlet problem.
    metrics = {
        "vertices": len(u),
        "triangles": len(cells),
        "free_dofs": len(free),
        "energy_error": float(np.sqrt(energy)),
        "l2_error": float(np.sqrt(l2)),
        "estimator": float(np.sqrt(eta2.sum())),
        "max_free_residual": float(np.max(np.abs(residual[free]), initial=0)),
        "global_balance_error": float(abs(residual[boundary].sum() + b.sum())),
    }
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("nonfinite solution")
    return Solution(mesh, u, flux, np.sqrt(eta2), metrics)


def mark(indicator, theta):
    if not 0 < theta <= 1:
        raise ValueError("bulk fraction must be in (0,1]")
    weights = indicator**2
    order = np.argsort(-weights, kind="stable")
    mask = np.zeros(len(weights), dtype=bool)
    if weights.sum() == 0:
        return mask
    split = (
        int(
            np.searchsorted(
                np.cumsum(weights[order]), theta * weights.sum(), side="left"
            )
        )
        + 1
    )
    mask[order[:split]] = True
    return mask


def run(
    case="corner", initial=4, steps=5, theta=0.55, mode="adaptive", max_cells=150000
):
    if (
        steps < 0
        or steps > 16
        or mode not in ("adaptive", "uniform")
        or case not in CASES
        or not 0 < theta <= 1
        or max_cells < 1
    ):
        raise ValueError("invalid refinement options")
    if case == "interface" and initial % 2:
        raise ValueError("interface requires even grid resolution")
    problem = CASES[case]
    mesh = Mesh.grid(initial, problem.domain)
    history = []
    if len(mesh.cells) > max_cells:
        raise ValueError("initial mesh exceeds the cell budget")
    for i in range(steps + 1):
        result = solve(mesh, problem)
        history.append(dict(step=i, **result.metrics))
        if i == steps or result.metrics["estimator"] < 1e-12:
            break
        flags = (
            mark(result.indicator, theta)
            if mode == "adaptive"
            else np.ones(len(mesh.cells), dtype=bool)
        )
        edges = np.unique(mesh.topology.local_edges[flags])
        # Every split edge adds one triangle for each incident old triangle.
        predicted = len(mesh.cells) + int(mesh.topology.counts[edges].sum())
        if predicted > max_cells:
            break
        mesh = mesh.refine(flags)
    return result, history
