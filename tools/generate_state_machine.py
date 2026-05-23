from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("report_figures/state_machine.png")
OUT.parent.mkdir(exist_ok=True)

states = {
    "BOOT": (0, 3),
    "IDLE": (2.4, 3),
    "ACQUIRE": (4.8, 3),
    "FILTER": (7.2, 3),
    "DETECT": (9.6, 3),
    "DISPLAY": (12, 3),
    "ERROR": (6, 0.7),
}

labels = {
    "BOOT": "BOOT\nInit peripherals",
    "IDLE": "IDLE\nWait valid ECG",
    "ACQUIRE": "ACQUIRE\nRead ADS1115",
    "FILTER": "FILTER\nDSP filtering",
    "DETECT": "DETECT\nR-peak, BPM, RR",
    "DISPLAY": "DISPLAY\nOLED / Serial / GUI",
    "ERROR": "ERROR\nLeads off / Noise",
}

fig, ax = plt.subplots(figsize=(15, 5))
ax.axis("off")
ax.set_xlim(-1, 13)
ax.set_ylim(0, 4.2)

def box(name):
    x, y = states[name]
    w, h = 1.8, 0.75
    p = FancyBboxPatch(
        (x - w/2, y - h/2),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.8,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(p)
    ax.text(x, y, labels[name], ha="center", va="center",
            fontsize=10, fontweight="bold")

def arrow(a, b, text="", curve=0):
    x1, y1 = states[a]
    x2, y2 = states[b]

    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.4,
        color="black",
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=42,
        shrinkB=42
    )
    ax.add_patch(arr)

    if text:
        ax.text(
            (x1+x2)/2,
            (y1+y2)/2 + 0.28,
            text,
            ha="center",
            va="center",
            fontsize=8
        )

for s in states:
    box(s)

arrow("BOOT", "IDLE", "init OK")
arrow("IDLE", "ACQUIRE", "valid leads")
arrow("ACQUIRE", "FILTER", "sample")
arrow("FILTER", "DETECT", "filtered")
arrow("DETECT", "DISPLAY", "metrics")
arrow("DISPLAY", "IDLE", "next cycle", curve=0.35)

arrow("ACQUIRE", "ERROR", "leads off", curve=-0.25)
arrow("FILTER", "ERROR", "bad quality", curve=-0.25)
arrow("DETECT", "ERROR", "invalid RR", curve=-0.25)
arrow("ERROR", "IDLE", "recover", curve=-0.25)

ax.text(
    6, 4.0,
    "Finite State Machine of the ECG Monitoring System",
    ha="center",
    fontsize=15,
    fontweight="bold"
)

ax.text(
    6, 0.15,
    "ESP32-S3: acquisition, filtering and R-peak detection | Python monitor: visualization and CNN-assisted analysis",
    ha="center",
    fontsize=9
)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {OUT}")