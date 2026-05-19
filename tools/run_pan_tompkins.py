import numpy as np
import matplotlib.pyplot as plt

from ecg_pan_tompkins import load_ecg_csv, pan_tompkins


CSV_FILE = "ecg_raw_capture.csv"


def main():
    ecg, fs = load_ecg_csv(CSV_FILE)

    print(f"Frequência de amostragem estimada: {fs:.2f} Hz")
    print(f"Amostras carregadas: {len(ecg)}")

    result = pan_tompkins(ecg, fs)

    print(f"Picos R detectados: {len(result['r_peaks'])}")
    print(f"BPM estimado: {result['bpm']:.1f}")

    if len(result["rr_intervals"]) > 0:
        print(f"RR médio: {np.mean(result['rr_intervals']):.3f} s")
        print(f"RR std: {np.std(result['rr_intervals']):.3f} s")

    t = np.arange(len(ecg)) / fs

    plt.figure(figsize=(13, 5))
    plt.plot(t, result["centered"], label="ECG bruto centralizado", alpha=0.4)
    plt.plot(t, result["filtered"], label="ECG filtrado 5–15 Hz", linewidth=2)

    if len(result["r_peaks"]) > 0:
        plt.scatter(
            result["r_peaks"] / fs,
            result["filtered"][result["r_peaks"]],
            color="red",
            marker="x",
            label="Picos R"
        )

    plt.title(f"Pan-Tompkins - Picos R detectados | BPM: {result['bpm']:.1f}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(13, 4))
    plt.plot(t, result["integrated"], label="Sinal integrado")
    plt.axhline(result["threshold"], linestyle="--", label="Limiar")
    plt.title("Sinal integrado - etapa final do Pan-Tompkins")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()