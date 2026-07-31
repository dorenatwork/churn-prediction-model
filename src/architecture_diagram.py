"""Dark-themed architecture diagram (matplotlib) for embedding in the PDF report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

BG = "#0d1117"
BOX_FACE = "#161b22"
BOX_EDGE = "#8b949e"
TEXT_COLOR = "#e6edf3"
ARROW_COLOR = "#6e7681"

NODES = {
    "raw": (1.6, 12.4, 2.6, 0.9, "Raw Data"),
    "prefect": (6.5, 12.4, 3.4, 0.9, "Prefect Orchestration"),
    "prep": (1.6, 10.6, 2.9, 0.9, "Data Preprocessing"),
    "mlflow": (6.7, 10.6, 2.9, 0.9, "MLflow Tracking"),
    "train": (3.6, 8.8, 3.0, 0.9, "Model Training"),
    "registry": (2.4, 7.0, 2.7, 0.9, "Model Registry"),
    "batch": (6.7, 7.0, 2.9, 0.9, "Batch Predictions"),
    "serving": (2.4, 5.2, 2.7, 0.9, "Model Serving"),
    "monitor": (6.7, 5.2, 3.1, 0.9, "Performance Monitoring"),
    "fastapi": (2.4, 3.4, 2.7, 0.9, "FastAPI Server"),
    "docker": (2.4, 1.9, 2.7, 0.9, "Docker Container"),
    "deploy": (2.4, 0.4, 2.7, 0.9, "Cloud Deployment"),
}

EDGES = [
    ("raw", "prep", 0),
    ("prefect", "prep", -0.3),
    ("prefect", "train", -0.25),
    ("prep", "train", 0.15),
    ("train", "mlflow", 0.25),
    ("train", "registry", -0.15),
    ("train", "batch", 0.2),
    ("prefect", "batch", 0.45),
    ("prefect", "monitor", 0.55),
    ("registry", "serving", 0),
    ("batch", "monitor", 0.15),
    ("serving", "fastapi", 0),
    ("fastapi", "docker", 0),
    ("docker", "deploy", 0),
]


def _center_bottom(n):
    x, y, w, h, _ = NODES[n]
    return x, y

def _pt(n, edge):
    x, y, w, h, _ = NODES[n]
    if edge == "bottom":
        return x, y
    return x, y + h


def run() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 13.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.3, 13.6)
    ax.axis("off")

    for key, (x, y, w, h, label) in NODES.items():
        box = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.1, edgecolor=BOX_EDGE, facecolor=BOX_FACE,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                 fontsize=10.5, color=TEXT_COLOR, fontweight="medium")

    for src, dst, rad in EDGES:
        x1, y1 = _pt(src, "bottom")
        w1 = NODES[src][2]
        x2, y2 = _pt(dst, "top")
        w2 = NODES[dst][2]
        start = (x1 + w1 / 2, y1)
        end = (x2 + w2 / 2, y2)
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="-|>", color=ARROW_COLOR, lw=1.3,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=2, shrinkB=2,
            ),
        )

    fig.tight_layout(pad=0.6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "architecture_diagram.png", dpi=150,
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'architecture_diagram.png'}")


if __name__ == "__main__":
    run()
