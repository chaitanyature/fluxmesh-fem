import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fluxmesh.solver import run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "laminated-plate.json"


class InputTests(unittest.TestCase):
    def test_linear_plate_without_heaters(self):
        spec = json.loads(EXAMPLE.read_text())
        spec.update(
            material={"x_breaks": [], "conductivities": [2.0]},
            boundary=[7.0, 2.0, -3.0],
            sources=[],
        )
        sol, _ = run(case=spec, steps=2)
        x, y = sol.mesh.points.T
        np.testing.assert_allclose(sol.values, 7 + 2 * x - 3 * y, atol=1e-12)
        np.testing.assert_allclose(
            sol.flux, np.tile([-4.0, 6.0], (len(sol.flux), 1)), atol=1e-12
        )
        self.assertNotIn("energy_error", sol.metrics)

    def test_moveable_bundle_and_material_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "plate.json"
            source.write_bytes(EXAMPLE.read_bytes())
            out = Path(tmp) / "solution"
            cmd = [sys.executable, "-m", "fluxmesh"]
            subprocess.run(
                [*cmd, "solve", source, "--steps", "2", "--out", out],
                stdout=subprocess.PIPE,
                check=True,
            )
            source.unlink()
            self.assertTrue(
                json.loads(subprocess.check_output([*cmd, "audit", out]))["valid"]
            )
        spec = json.loads(EXAMPLE.read_text())
        spec["material"]["x_breaks"] = [0.48]
        with self.assertRaisesRegex(ValueError, "align"):
            run(case=spec)
        spec["material"]["x_breaks"] = [0.5]
        spec["sources"][0]["sigma"] = 0
        with self.assertRaisesRegex(ValueError, "sigma"):
            run(case=spec)
