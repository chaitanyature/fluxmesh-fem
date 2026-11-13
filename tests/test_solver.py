import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fluxmesh.artifacts import audit, bundle
from fluxmesh.cases import CASES, Problem, one, zero
from fluxmesh.mesh import Mesh
from fluxmesh.solver import solve, run, mark


class MeshChecks(unittest.TestCase):
    def test_random_refinement_conserves_area_and_is_conforming(self):
        rng = np.random.default_rng(373)
        for domain, area in [("square", 1.0), ("corner", 3.0)]:
            mesh = Mesh.grid(4, domain)
            for _ in range(4):
                mask = rng.random(len(mesh.cells)) < 0.25
                old = len(mesh.cells)
                edges = np.unique(mesh.topology.local_edges[mask])
                expected = old + mesh.topology.counts[edges].sum()
                mesh = mesh.refine(mask)
                self.assertEqual(len(mesh.cells), expected)
                self.assertAlmostEqual(mesh.area.sum(), area)
                self.assertEqual(
                    len(mesh.points) - len(mesh.topology.edges) + len(mesh.cells), 1
                )
                boundary_len = mesh.length[mesh.topology.counts == 1].sum()
                self.assertAlmostEqual(boundary_len, 4.0 if domain == "square" else 8.0)
                self.assertTrue((mesh.area > 0).all())

    def test_bulk_set_is_minimal_and_stable(self):
        eta = np.array([1.0, 2.0, 2.0, 3.0, 0.5])
        mask = mark(eta, 0.68)
        total = np.sum(eta**2)
        self.assertGreaterEqual(np.sum(eta[mask] ** 2), 0.68 * total)
        self.assertLess(np.sum(eta[mask] ** 2) - min(eta[mask]) ** 2, 0.68 * total)
        self.assertEqual(np.flatnonzero(mask).tolist(), [1, 3])
        with self.assertRaises(ValueError):
            mark(eta, 0)

    def test_affine_patch_under_local_refinement(self):
        def exact(x):
            return 1 + 2 * x[..., 0] - 3 * x[..., 1]

        def grad(x):
            return np.broadcast_to([2.0, -3.0], x.shape)

        prob = Problem("affine", "square", one, zero, exact, grad)
        mesh = Mesh.grid(4)
        mesh = mesh.refine(np.arange(len(mesh.cells)) % 3 == 0)
        result = solve(mesh, prob)
        np.testing.assert_allclose(result.values, exact(mesh.points), atol=5e-13)
        np.testing.assert_allclose(
            result.flux, np.tile([-2.0, 3.0], (len(mesh.cells), 1)), atol=5e-12
        )
        self.assertLess(result.metrics["estimator"], 1e-12)

    def test_exact_flux_transmission_patch(self):
        sol, history = run("interface", steps=5)
        self.assertEqual(len(history), 1)
        np.testing.assert_allclose(
            sol.values, CASES["interface"].boundary(sol.mesh.points), atol=1e-12
        )
        self.assertLess(abs(sol.flux[:, 0] + 1).max(), 1e-12)
        with self.assertRaisesRegex(ValueError, "even"):
            run("interface", initial=3)

    def test_smooth_rates_and_residual(self):
        sol, rows = run("smooth", steps=3, mode="uniform")
        rates = [
            np.log2(rows[-2][k] / rows[-1][k]) for k in ["energy_error", "l2_error"]
        ]
        self.assertGreater(rates[0], 0.96)
        self.assertGreater(rates[1], 1.9)
        self.assertLess(sol.metrics["global_balance_error"], 1e-10)

    def test_corner_adaptivity_beats_larger_uniform_mesh(self):
        adaptive, h = run("corner", steps=7)
        uniform, _ = run("corner", steps=4, mode="uniform")
        self.assertLess(len(adaptive.mesh.cells), len(uniform.mesh.cells) / 2)
        self.assertLess(
            adaptive.metrics["energy_error"], uniform.metrics["energy_error"] * 0.8
        )
        self.assertTrue(
            all(b["energy_error"] < a["energy_error"] for a, b in zip(h, h[1:]))
        )
        self.assertLess(adaptive.metrics["max_free_residual"], 1e-10)

    def test_cell_budget_honored_before_refinement(self):
        sol, h = run("corner", max_cells=43, steps=10)
        self.assertEqual(len(h), 2)
        self.assertLessEqual(len(sol.mesh.cells), 43)
        with self.assertRaises(ValueError):
            run("corner", max_cells=10)

    def test_invalid_mesh_and_unaligned_interface(self):
        with self.assertRaisesRegex(ValueError, "orientation"):
            Mesh([[0, 0], [0, 1], [1, 0]], [[0, 1, 2]])
        with self.assertRaisesRegex(ValueError, "interfaces"):
            solve(Mesh.grid(3), CASES["interface"])

    def test_bundle_audit_detects_changed_interior_values(self):
        sol, h = run("smooth", steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            bundle(out, sol, h, {"case": "smooth"})
            self.assertTrue(audit(out)["valid"])
            self.assertIn("DATASET UNSTRUCTURED_GRID", (out / "field.vtk").read_text())
            with self.assertRaises(FileExistsError):
                bundle(out, sol, h, {"case": "smooth"})
            with np.load(out / "field.npz") as data:
                arrays = dict(data)
            interior = np.setdiff1d(np.arange(len(sol.values)), sol.mesh.boundary)
            arrays["values"][interior[0]] += 0.5
            np.savez(out / "field.npz", **arrays)
            self.assertFalse(audit(out)["valid"])

    def test_public_cli_to_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case"
            cmd = [sys.executable, "-m", "fluxmesh"]
            subprocess.run(
                [*cmd, "solve", "corner", "--steps", "2", "--out", path],
                check=True,
                stdout=subprocess.PIPE,
            )
            self.assertTrue(
                json.loads(subprocess.check_output([*cmd, "audit", path]))["valid"]
            )
