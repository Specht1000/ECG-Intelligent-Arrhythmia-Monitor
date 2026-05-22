import sys
import time
import queue
import threading
from pathlib import Path
from collections import deque, Counter

import serial
import numpy as np
import pandas as pd
import joblib
import tkinter as tk
from tkinter import ttk
from serial.tools import list_ports
from tensorflow.keras.models import load_model

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


BAUD = 115200
FS = 250
MAX_POINTS = 1200

CNN_BEAT_LEN = 252
CONF_THRESHOLD = 0.70
CNN_INTERVAL_SEC = 1.0


def get_root():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


ROOT = get_root()
MODEL_DIR = ROOT / "models" / "trained"

CNN_FILE = MODEL_DIR / "heartbeat_cnn.keras"

ENCODER_CANDIDATES = [
    MODEL_DIR / "heartbeat_label_encoder.pkl",
    MODEL_DIR / "label_encoder_mitdb.pkl",
]

QUALITY_CANDIDATES = [
    MODEL_DIR / "signal_quality_model.pkl",
    MODEL_DIR / "signal_quality_model.joblib",
]


class RealtimeHonestMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("PFE ECG Monitor - Realtime AI Monitor")
        self.root.geometry("1500x900")

        self.serial_port = None
        self.running = False
        self.q = queue.Queue()

        self.model = None
        self.encoder = None
        self.quality_model = None

        self.raw_samples = []
        self.filtered_samples = []

        self.ecg_buffer = deque(maxlen=MAX_POINTS)
        self.peak_buffer = deque(maxlen=MAX_POINTS)
        self.raw_quality_buffer = deque(maxlen=FS * 5)
        self.ai_history = deque(maxlen=30)

        self.bpm = 0.0
        self.rr = 0.0
        self.raw = 0
        self.filtered = 0.0
        self.integrated = 0.0
        self.threshold = 0.0
        self.status = "WAITING"
        self.signal_quality = "WAITING"
        self.ai_class = "WAITING"
        self.ai_confidence = 0.0
        self.ai_status = "WAITING"
        self.stream = "WAITING"

        self.last_plot_time = 0.0
        self.last_cnn_time = 0.0
        self.last_quality_time = 0.0

        self.build_ui()
        self.load_models()
        self.refresh_ports()
        self.update_loop()

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=6)

        tk.Label(top, text="Port:").pack(side="left")
        self.port_combo = ttk.Combobox(top, width=12)
        self.port_combo.pack(side="left", padx=5)

        tk.Button(top, text="Refresh", command=self.refresh_ports).pack(side="left", padx=5)
        tk.Button(top, text="Connect", command=self.connect).pack(side="left", padx=5)
        tk.Button(top, text="Disconnect", command=self.disconnect).pack(side="left", padx=5)

        info = tk.LabelFrame(self.root, text="Medical information + CNN")
        info.pack(fill="x", padx=10, pady=6)

        self.bpm_label = tk.Label(info, text="BPM: 0.0", font=("Arial", 28, "bold"))
        self.bpm_label.grid(row=0, column=0, padx=25, pady=6)

        self.rr_label = tk.Label(info, text="RR: 0.000 s", font=("Arial", 20))
        self.rr_label.grid(row=0, column=1, padx=25, pady=6)

        self.status_label = tk.Label(info, text="AI STATUS: WAITING", font=("Arial", 22, "bold"))
        self.status_label.grid(row=0, column=2, padx=25, pady=6)

        self.ai_label = tk.Label(info, text="CNN class: WAITING", font=("Arial", 18, "bold"))
        self.ai_label.grid(row=1, column=0, padx=25, pady=6)

        self.conf_label = tk.Label(info, text="Confidence: 0.00", font=("Arial", 18))
        self.conf_label.grid(row=1, column=1, padx=25, pady=6)

        self.quality_label = tk.Label(info, text="Signal quality: WAITING", font=("Arial", 18, "bold"))
        self.quality_label.grid(row=1, column=2, padx=25, pady=6)

        self.tech_label = tk.Label(info, text="RAW: 0 | FILT: 0.0 | INT: 0.0 | TH: 0.0", font=("Arial", 15))
        self.tech_label.grid(row=2, column=0, columnspan=2, padx=25, pady=6)

        self.stream_label = tk.Label(info, text="STREAM: WAITING", font=("Arial", 15))
        self.stream_label.grid(row=2, column=2, padx=25, pady=6)

        graph = tk.LabelFrame(self.root, text="Realtime ECG")
        graph.pack(fill="both", expand=True, padx=10, pady=6)

        self.fig = Figure(figsize=(12, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_title("Filtered ECG signal")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")

        self.line, = self.ax.plot([], [], color="black", linewidth=1.2)
        self.peaks_plot, = self.ax.plot([], [], "rx", markersize=8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        term_frame = tk.LabelFrame(self.root, text="Serial terminal")
        term_frame.pack(fill="both", padx=10, pady=6)

        self.terminal = tk.Text(
            term_frame,
            height=9,
            font=("Consolas", 10),
            bg="black",
            fg="#00ff66",
            insertbackground="white"
        )
        self.terminal.pack(side="left", fill="both", expand=True)

        scroll = tk.Scrollbar(term_frame, command=self.terminal.yview)
        scroll.pack(side="right", fill="y")
        self.terminal.config(yscrollcommand=scroll.set)

    def load_models(self):
        try:
            self.model = load_model(CNN_FILE)
            self.log(f"[AI] CNN loaded: {CNN_FILE.name}")
        except Exception as e:
            self.log(f"[ERROR] CNN not loaded: {e}")

        for f in ENCODER_CANDIDATES:
            if f.exists():
                try:
                    self.encoder = joblib.load(f)
                    self.log(f"[AI] Encoder loaded: {f.name}")
                    break
                except Exception as e:
                    self.log(f"[ERROR] Encoder error: {e}")

        for f in QUALITY_CANDIDATES:
            if f.exists():
                try:
                    self.quality_model = joblib.load(f)
                    self.log(f"[AI] Quality model loaded: {f.name}")
                    break
                except Exception as e:
                    self.log(f"[ERROR] Quality model error: {e}")

    def refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        self.port_combo["values"] = ports

        if "COM7" in ports:
            self.port_combo.set("COM7")
        elif "COM9" in ports:
            self.port_combo.set("COM9")
        elif ports:
            self.port_combo.set(ports[0])

    def connect(self):
        port = self.port_combo.get()

        try:
            self.serial_port = serial.Serial(
                port,
                BAUD,
                timeout=0.05,
                write_timeout=0.05
            )

            self.running = True

            threading.Thread(target=self.serial_worker, daemon=True).start()
            threading.Thread(target=self.ai_mode_worker, daemon=True).start()

            self.ai_status = "CONNECTED"
            self.log(f"[SYSTEM] Connected to {port}")
            self.log("[SYSTEM] Requesting MODE_AI...")

        except Exception as e:
            self.log(f"[ERROR] Connection failed: {e}")

    def ai_mode_worker(self):
        time.sleep(0.5)

        while self.running:
            try:
                self.serial_port.write(b"MODE_AI\n")
                self.serial_port.write(b"DATA\n")
                self.serial_port.flush()
                self.q.put("[SYSTEM] MODE_AI sent")
            except Exception as e:
                self.q.put(f"[ERROR] MODE_AI failed: {e}")

            time.sleep(1.0)

    def disconnect(self):
        self.running = False

        if self.serial_port:
            try:
                self.serial_port.write(b"MODE_DOCTOR\n")
                self.serial_port.flush()
                time.sleep(0.1)
                self.serial_port.close()
            except Exception:
                pass

        self.ai_status = "DISCONNECTED"
        self.stream = "OFF"
        self.log("[SYSTEM] Disconnected")

    def serial_worker(self):
        while self.running:
            try:
                line = self.serial_port.readline().decode(errors="ignore").strip()
                if line:
                    self.q.put(line)
            except Exception as e:
                self.q.put(f"[ERROR] Serial error: {e}")

    def parse_line(self, line):
        if line.startswith("[SYSTEM]"):
            self.log(line)
            return

        if line.startswith("[ERROR]"):
            self.log(line)
            return

        if "LEADS_OFF" in line:
            self.force_leads_off()
            self.log(line)
            return

        if line.startswith("DATA,"):
            self.parse_data(line)
            return

        if line.startswith("[ECG]"):
            self.parse_ecg_log(line)
            return

        if line.startswith("[RPEAK]"):
            self.parse_rpeak_log(line)
            return

        if line.startswith("[SERIAL]") or line.startswith("[BOOT]"):
            self.log(line)

    def parse_data(self, line):
        try:
            p = line.split(",")

            if len(p) < 9:
                return

            self.stream = "DATA"

            self.raw = int(float(p[2]))
            self.filtered = float(p[3])
            self.integrated = float(p[4])
            r_peak = int(p[5])
            self.bpm = float(p[6])
            self.threshold = float(p[7])
            self.status = p[8].strip()

            self.rr = 60.0 / self.bpm if self.bpm > 0 else 0.0

            display_signal = (self.raw - 20000) * 4.0
            self.add_sample(self.raw, display_signal, r_peak)

            now = time.time()

            if now - self.last_quality_time >= 1.0:
                self.last_quality_time = now
                self.update_signal_quality()

            if now - self.last_cnn_time >= CNN_INTERVAL_SEC:
                self.last_cnn_time = now
                self.run_cnn_continuous()

            self.update_ai_status()

        except Exception as e:
            self.log(f"[ERROR] DATA parse error: {e}")

    def parse_ecg_log(self, line):
        try:
            self.stream = "LOG"

            if "RAW=" in line:
                self.raw = int(float(line.split("RAW=")[1].split("|")[0].strip()))

            if "FILT=" in line:
                self.filtered = float(line.split("FILT=")[1].split("|")[0].strip())

            if "INT=" in line:
                self.integrated = float(line.split("INT=")[1].split("|")[0].strip())

            if "TH=" in line:
                self.threshold = float(line.split("TH=")[1].split("|")[0].strip())

            if "BPM=" in line:
                self.bpm = float(line.split("BPM=")[1].split("|")[0].strip())

            if "STATUS=" in line:
                self.status = line.split("STATUS=")[1].strip()

            self.rr = 60.0 / self.bpm if self.bpm > 0 else 0.0

            self.add_sample(self.raw, self.filtered, 0)

        except Exception:
            pass

    def parse_rpeak_log(self, line):
        try:
            self.stream = "LOG"

            if "RR=" in line:
                self.rr = float(line.split("RR=")[1].split("|")[0].strip())

            if "BPM_SMOOTH=" in line:
                self.bpm = float(line.split("BPM_SMOOTH=")[1].split("|")[0].strip())
            elif "BPM=" in line:
                self.bpm = float(line.split("BPM=")[1].split("|")[0].strip())

            if "STATUS=" in line:
                self.status = line.split("STATUS=")[1].strip()

        except Exception:
            pass

    def add_sample(self, raw, filtered, peak):
        self.raw_samples.append(raw)
        self.filtered_samples.append(filtered)

        self.raw_quality_buffer.append(raw)
        self.ecg_buffer.append(filtered)
        self.peak_buffer.append(peak)

    def force_leads_off(self):
        self.status = "LEADS_OFF"
        self.signal_quality = "BAD"
        self.ai_status = "LEADS OFF"
        self.ai_class = "WAITING"
        self.ai_confidence = 0.0

        self.bpm = 0.0
        self.rr = 0.0
        self.filtered = 0.0
        self.integrated = 0.0
        self.threshold = 0.0

    def update_signal_quality(self):
        if self.quality_model is None:
            if len(self.raw_quality_buffer) >= FS:
                raw = np.array(self.raw_quality_buffer, dtype=np.float32)
                if np.std(raw) < 20:
                    self.signal_quality = "bad"
                elif np.std(raw) < 120:
                    self.signal_quality = "acceptable"
                else:
                    self.signal_quality = "good"
            return

        if len(self.raw_quality_buffer) < FS * 5:
            return

        raw = np.array(self.raw_quality_buffer, dtype=np.float32)
        diff = np.diff(raw)

        features = {
            "std": np.std(raw),
            "range": np.max(raw) - np.min(raw),
            "diff_std": np.std(diff),
            "energy": np.sum((raw - np.mean(raw)) ** 2) / len(raw),
            "zero_or_clip": np.sum((raw <= 10) | (raw >= 32760)),
        }

        try:
            names = getattr(self.quality_model, "feature_names_in_", list(features.keys()))
            X = pd.DataFrame(
                [{k: features.get(k, 0.0) for k in names}],
                columns=names
            )
            self.signal_quality = str(self.quality_model.predict(X)[0])
        except Exception as e:
            self.signal_quality = "UNKNOWN"
            self.log(f"[ERROR] Quality predict error: {e}")

    def run_cnn_continuous(self):
        if self.model is None or self.encoder is None:
            return

        if len(self.filtered_samples) < CNN_BEAT_LEN:
            return

        try:
            beat = np.array(
                self.filtered_samples[-CNN_BEAT_LEN:],
                dtype=np.float32
            )

            beat = beat - np.mean(beat)
            std = np.std(beat)

            if std > 1e-6:
                beat = beat / std

            x = beat.reshape(1, CNN_BEAT_LEN, 1)

            pred = self.model.predict(x, verbose=0)[0]

            idx = int(np.argmax(pred))
            label = str(self.encoder.inverse_transform([idx])[0])
            conf = float(pred[idx])

            final = label if conf >= CONF_THRESHOLD else "low_confidence"

            self.ai_class = final
            self.ai_confidence = conf
            self.ai_history.append(final)

            self.log(f"[CNN] class={label} | final={final} | confidence={conf:.3f}")

        except Exception as e:
            self.log(f"[ERROR] CNN predict error: {e}")

    def update_ai_status(self):
        if self.status == "LEADS_OFF":
            self.ai_status = "LEADS OFF"
            return

        if str(self.signal_quality).lower() == "bad":
            self.ai_status = "POOR SIGNAL"
            return

        valid = [
            x for x in self.ai_history
            if x not in ["low_confidence", "WAITING", None]
        ]

        if len(valid) < 3:
            self.ai_status = "ANALYZING"
            return

        c = Counter(valid)
        total = len(valid)

        if c.get("ventricular", 0) / total >= 0.15:
            self.ai_status = "VENTRICULAR ALERT"
        elif c.get("supraventricular", 0) / total >= 0.25:
            self.ai_status = "SUPRAVENTRICULAR"
        else:
            self.ai_status = "NORMAL"

    def update_plot(self):
        if len(self.ecg_buffer) < 5:
            return

        y = np.array(self.ecg_buffer, dtype=np.float32)
        x = np.arange(len(y))

        self.line.set_data(x, y)

        peak_x = []
        peak_y = []

        peaks = list(self.peak_buffer)

        for i, p in enumerate(peaks):
            if p == 1 and i < len(y):
                peak_x.append(i)
                peak_y.append(y[i])

        self.peaks_plot.set_data(peak_x, peak_y)

        current_len = len(y)

        VIEW_SAMPLES = 750

        if current_len < VIEW_SAMPLES:
            xmin = 0
            xmax = VIEW_SAMPLES
        else:
            xmin = current_len - VIEW_SAMPLES
            xmax = current_len

        self.ax.set_xlim(xmin, xmax)

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        margin = max(300.0, (y_max - y_min) * 0.25)

        self.ax.set_ylim(y_min - margin, y_max + margin)

        self.canvas.draw_idle()

    def update_loop(self):
        processed = 0

        while not self.q.empty() and processed < 40:
            try:
                line = self.q.get_nowait()
                self.parse_line(line)
                processed += 1
            except Exception as e:
                self.log(f"[ERROR] update loop: {e}")
                break

        self.bpm_label.config(text=f"BPM: {self.bpm:.1f}")
        self.rr_label.config(text=f"RR: {self.rr:.3f} s")
        self.status_label.config(text=f"AI STATUS: {self.ai_status}")
        self.ai_label.config(text=f"CNN class: {self.ai_class}")
        self.conf_label.config(text=f"Confidence: {self.ai_confidence:.2f}")
        self.quality_label.config(text=f"Signal quality: {self.signal_quality}")
        self.stream_label.config(text=f"STREAM: {self.stream}")
        self.tech_label.config(
            text=(
                f"RAW: {self.raw} | "
                f"FILT: {self.filtered:.1f} | "
                f"INT: {self.integrated:.1f} | "
                f"TH: {self.threshold:.1f}"
            )
        )

        now = time.time()

        if now - self.last_plot_time > 0.15:
            self.last_plot_time = now
            self.update_plot()

        self.root.after(40, self.update_loop)

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")

        self.terminal.insert("end", f"[{timestamp}] {msg}\n")

        lines = int(self.terminal.index("end-1c").split(".")[0])

        if lines > 250:
            self.terminal.delete("1.0", "50.0")

        self.terminal.see("end")


def main():
    root = tk.Tk()
    RealtimeHonestMonitor(root)
    root.mainloop()


if __name__ == "__main__":
    main()