from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Topology:
    edges: np.ndarray
    local_edges: np.ndarray
    counts: np.ndarray
    owners: np.ndarray


class Mesh:
    """Counterclockwise triangles. One global edge numbering drives both sides of a split."""

    def __init__(self, points, cells):
        self.points = np.array(points, dtype=float, copy=True)
        self.cells = np.array(cells, dtype=np.int64, copy=True)
        if (
            self.points.ndim != 2
            or self.points.shape[1] != 2
            or not np.isfinite(self.points).all()
        ):
            raise ValueError("expected finite 2D vertices")
        if self.cells.ndim != 2 or self.cells.shape[1] != 3 or not len(self.cells):
            raise ValueError("expected triangular cells")
        if self.cells.min() < 0 or self.cells.max() >= len(self.points):
            raise ValueError("invalid vertex index")
        xyz = self.points[self.cells]
        u, v = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
        twice = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        if np.any(twice <= 0):
            raise ValueError("triangle orientation or area invalid")
        self.area = twice / 2
        # grad(phi_i) is the inward normal to the opposing edge divided by twice the area.
        sides = np.roll(xyz, -1, axis=1) - np.roll(xyz, -2, axis=1)
        self.grad = (
            np.stack((sides[:, :, 1], -sides[:, :, 0]), axis=2) / twice[:, None, None]
        )
        ends = np.sort(
            np.stack((self.cells, np.roll(self.cells, -1, axis=1)), axis=-1), axis=2
        )
        edges, inv, counts = np.unique(
            ends.reshape(-1, 2), axis=0, return_inverse=True, return_counts=True
        )
        if counts.max() > 2:
            raise ValueError("non-manifold edge")
        local = inv.reshape(-1, 3)
        owners = np.full((len(edges), 2), -1, dtype=int)
        order = np.argsort(local.ravel(), kind="stable")
        offset = np.r_[0, counts.cumsum()[:-1]]
        owners[:, 0] = order[offset] // 3
        interior = counts == 2
        owners[interior, 1] = order[offset[interior] + 1] // 3
        self.topology = Topology(edges, local, counts, owners)
        self.edge_vectors = self.points[edges[:, 1]] - self.points[edges[:, 0]]
        self.length = np.linalg.norm(self.edge_vectors, axis=1)
        self.diameter = self.length[local].max(axis=1)
        self.boundary = np.unique(edges[~interior])
        for array in (
            self.cells,
            self.points,
            self.area,
            self.grad,
            edges,
            local,
            counts,
            owners,
            self.edge_vectors,
            self.length,
            self.diameter,
            self.boundary,
        ):
            array.setflags(write=False)

    @classmethod
    def grid(cls, n=4, domain="square"):
        if type(n) is not int or n < 2 or n > 400 or domain not in ("square", "corner"):
            raise ValueError("invalid mesh grid")
        if domain == "corner" and n % 2:
            raise ValueError("corner grids need even n")
        axis = np.linspace(-1 if domain == "corner" else 0, 1, n + 1)
        points = []
        ids = {}
        cells = []

        def vertex(i, j):
            if (i, j) not in ids:
                ids[i, j] = len(points)
                points.append((axis[i], axis[j]))
            return ids[i, j]

        for j in range(n):
            for i in range(n):
                if domain == "corner" and i >= n // 2 and j < n // 2:
                    continue
                a, b, c, d = [
                    vertex(x, y)
                    for x, y in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
                ]
                cells.extend(
                    [(a, b, c), (a, c, d)]
                    if (i + j) % 2 == 0
                    else [(a, b, d), (b, c, d)]
                )
        return cls(points, cells)

    def refine(self, marked):
        marked = np.asarray(marked)
        if marked.dtype != bool or marked.shape != (len(self.cells),):
            raise ValueError("invalid cell mask")
        if not marked.any():
            return self
        t = self.topology
        split = np.unique(t.local_edges[marked])
        middle = np.full(len(t.edges), -1, dtype=int)
        middle[split] = len(self.points) + np.arange(len(split))
        pts = np.vstack((self.points, self.points[t.edges[split]].mean(axis=1)))
        children = []
        for cell, ids in zip(self.cells, t.local_edges):
            v = list(cell)
            m = list(middle[ids])
            count = sum(k >= 0 for k in m)
            if count == 0:
                children.append(v)
                continue
            if count == 3:
                a, b, c = v
                x, y, z = m
                children.extend([(a, x, z), (x, b, y), (z, y, c), (x, y, z)])
                continue
            # Rotate both lists in the same order; preserve positive orientation.
            start = (
                next(i for i in range(3) if m[i] >= 0)
                if count == 1
                else (m.index(-1) + 1) % 3
            )
            a, b, c = v[start:] + v[:start]
            x, y, _ = m[start:] + m[:start]
            if count == 1:
                children.extend([(a, x, c), (x, b, c)])
            else:
                children.append((x, b, y))
                if np.linalg.norm(pts[a] - pts[y]) <= np.linalg.norm(pts[x] - pts[c]):
                    children.extend([(a, x, y), (a, y, c)])
                else:
                    children.extend([(a, x, c), (x, y, c)])
        return Mesh(pts, children)
