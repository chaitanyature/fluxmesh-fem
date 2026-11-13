from .solver import run


def compare(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    adaptive, ha = run("corner", steps=7)
    _, hu = run("corner", steps=4, mode="uniform")
    ink = "#1b3440"
    teal = "#007d78"
    orange = "#b75729"
    paper = "#faf8ee"
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": paper,
            "figure.facecolor": paper,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11.2, 5),
            layout="constrained",
            gridspec_kw={"width_ratios": [1, 1.15]},
        )
        ax, curve = axes
        m = adaptive.mesh
        tri = Triangulation(*m.points.T, triangles=m.cells)
        ax.tripcolor(tri, adaptive.values, cmap="viridis", shading="gouraud")
        ax.triplot(tri, color="white", lw=0.19, alpha=0.67)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Mesh near the re-entrant corner", loc="left", fontsize=13)
        for rows, label, color, mrk in [
            (hu, "Uniform", orange, "s"),
            (ha, "Adaptive", teal, "o"),
        ]:
            curve.loglog(
                [r["triangles"] for r in rows],
                [r["energy_error"] for r in rows],
                marker=mrk,
                color=color,
                lw=2.3,
                label=label,
            )
        curve.legend(frameon=False)
        curve.grid(which="major", color="#bfc9c4", lw=0.7)
        curve.set_xlabel("Triangle count")
        curve.set_ylabel("Energy-norm error")
        curve.set_title("Accuracy at each refinement level", loc="left", fontsize=13)
        for side in ("top", "right"):
            curve.spines[side].set_visible(False)
        fig.suptitle(
            "FluxMesh   /   Corner singularity",
            fontsize=23,
            fontweight="bold",
            ha="left",
            x=0.02,
            color=ink,
        )
        fig.savefig(path, dpi=145)
        plt.close(fig)
    return {"adaptive": ha[-1], "uniform": hu[-1]}
