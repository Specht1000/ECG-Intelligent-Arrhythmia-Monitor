from pathlib import Path
from graphviz import Digraph

OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

dot = Digraph("PFE_ECG_SYSTEM")

dot.attr(rankdir="LR")
dot.attr(fontsize="20")

dot.node("E", "Eletrodos ECG")
dot.node("A", "AD8232\nFront-end analógico")
dot.node("D", "ADS1115\nADC 16 bits")
dot.node("ESP", "ESP32-S3")

dot.node("P", "Pré-processamento")
dot.node("R", "Detecção de picos R")
dot.node("B", "Estimativa BPM")
dot.node("CNN", "CNN 1D\nClassificação")
dot.node("Q", "Qualidade do sinal")
dot.node("C", "Decisão clínica")

dot.node("OUT", "Monitor e alertas")

dot.edge("E", "A")
dot.edge("A", "D")
dot.edge("D", "ESP", label="I2C")

dot.edge("ESP", "P")
dot.edge("P", "R")
dot.edge("R", "B")
dot.edge("R", "CNN")

dot.edge("P", "Q")

dot.edge("CNN", "C")
dot.edge("B", "C")
dot.edge("Q", "C")

dot.edge("C", "OUT")

output = OUT_DIR / "system_architecture"

dot.render(str(output), format="png", cleanup=True)

print(f"Figura salva em: {output}.png")