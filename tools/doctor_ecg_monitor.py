import time
import threading
import serial
import tkinter as tk
from tkinter import ttk
from serial.tools import list_ports


BAUD = 115200


class DoctorECGTerminal:
    def __init__(self, root):
        self.root = root
        self.root.title("PFE ECG Monitor - Medical Terminal")
        self.root.geometry("1200x750")

        self.serial_port = None
        self.running = False

        self.bpm = 0.0
        self.rr = 0.0
        self.status = "WAITING"
        self.signal_quality = "WAITING"
        self.raw = 0
        self.filtered = 0.0
        self.integrated = 0.0
        self.threshold = 0.0
        self.stream_mode = "WAITING"

        self.build_ui()
        self.refresh_ports()
        self.update_labels()

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=8)

        tk.Label(top, text="Port:").pack(side="left")

        self.port_combo = ttk.Combobox(top, width=12)
        self.port_combo.pack(side="left", padx=5)

        tk.Button(top, text="Refresh", command=self.refresh_ports).pack(side="left", padx=5)
        tk.Button(top, text="Connect", command=self.connect).pack(side="left", padx=5)
        tk.Button(top, text="Disconnect", command=self.disconnect).pack(side="left", padx=5)

        info = tk.LabelFrame(self.root, text="Medical information")
        info.pack(fill="x", padx=10, pady=8)

        self.bpm_label = tk.Label(info, text="BPM: 0.0", font=("Arial", 28, "bold"))
        self.bpm_label.grid(row=0, column=0, padx=30, pady=8)

        self.rr_label = tk.Label(info, text="RR: 0.000 s", font=("Arial", 20))
        self.rr_label.grid(row=0, column=1, padx=30, pady=8)

        self.status_label = tk.Label(info, text="STATUS: WAITING", font=("Arial", 22, "bold"))
        self.status_label.grid(row=0, column=2, padx=30, pady=8)

        self.quality_label = tk.Label(info, text="Signal quality: WAITING", font=("Arial", 18, "bold"))
        self.quality_label.grid(row=1, column=0, padx=30, pady=8)

        self.stream_label = tk.Label(info, text="STREAM: WAITING", font=("Arial", 18))
        self.stream_label.grid(row=1, column=1, padx=30, pady=8)

        self.tech_label = tk.Label(info, text="RAW: 0 | FILT: 0.0 | INT: 0.0 | TH: 0.0", font=("Arial", 15))
        self.tech_label.grid(row=1, column=2, padx=30, pady=8)

        terminal_frame = tk.LabelFrame(self.root, text="Serial terminal")
        terminal_frame.pack(fill="both", expand=True, padx=10, pady=8)

        self.terminal = tk.Text(
            terminal_frame,
            font=("Consolas", 11),
            bg="black",
            fg="#00ff66",
            insertbackground="white"
        )
        self.terminal.pack(side="left", fill="both", expand=True)

        scroll = tk.Scrollbar(terminal_frame, command=self.terminal.yview)
        scroll.pack(side="right", fill="y")
        self.terminal.config(yscrollcommand=scroll.set)

    def refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        self.port_combo["values"] = ports

        if "COM9" in ports:
            self.port_combo.set("COM9")
        elif ports:
            self.port_combo.set(ports[0])

    def connect(self):
        port = self.port_combo.get()

        if not port:
            self.log("[ERROR] No COM port selected")
            return

        try:
            self.serial_port = serial.Serial(port, BAUD, timeout=1)
            self.running = True

            threading.Thread(target=self.serial_worker, daemon=True).start()
            threading.Thread(target=self.force_stream_on, daemon=True).start()

            self.status = "CONNECTED"
            self.log(f"[SYSTEM] Connected to {port}")

        except Exception as e:
            self.log(f"[ERROR] Connection failed: {e}")

    def disconnect(self):
        self.running = False

        if self.serial_port:
            try:
                self.serial_port.write(b"STREAM_OFF\n")
                time.sleep(0.2)
                self.serial_port.close()
            except Exception:
                pass

        self.serial_port = None
        self.status = "DISCONNECTED"
        self.stream_mode = "OFF"
        self.log("[SYSTEM] Disconnected")

    def force_stream_on(self):
        time.sleep(1.5)

        while self.running:
            try:
                self.serial_port.write(b"MODE_DOCTOR\n")
            except Exception:
                pass

            time.sleep(2)

    def serial_worker(self):
        while self.running:
            try:
                line = self.serial_port.readline().decode(errors="ignore").strip()

                if not line:
                    continue

                self.parse_line(line)
                self.log(line)

            except Exception as e:
                self.log(f"[ERROR] Serial error: {e}")

    def parse_line(self, line):
        if "LEADS_OFF" in line:
            self.force_leads_off(line)
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

    def force_leads_off(self, line):
        self.status = "LEADS_OFF"
        self.signal_quality = "BAD"
        self.bpm = 0.0
        self.rr = 0.0
        self.filtered = 0.0
        self.integrated = 0.0
        self.threshold = 0.0

        if line.startswith("DATA,"):
            self.stream_mode = "DATA"
            try:
                parts = line.split(",")
                self.raw = int(float(parts[2]))
            except Exception:
                pass
        else:
            self.stream_mode = "LOG"
            try:
                if "RAW=" in line:
                    self.raw = int(float(line.split("RAW=")[1].split("|")[0].strip()))
            except Exception:
                pass

    def parse_data(self, line):
        try:
            parts = line.split(",")

            self.stream_mode = "DATA"

            self.raw = int(float(parts[2]))
            self.filtered = float(parts[3])
            self.integrated = float(parts[4])
            self.bpm = float(parts[6])
            self.threshold = float(parts[7])
            self.status = parts[8].strip()

            if self.status == "LEADS_OFF":
                self.force_leads_off(line)
                return

            if self.bpm > 0:
                self.rr = 60.0 / self.bpm
            else:
                self.rr = 0.0

            self.update_quality_from_status()

        except Exception as e:
            self.log(f"[ERROR] DATA parse error: {e}")

    def parse_ecg_log(self, line):
        try:
            self.stream_mode = "LOG"

            if "RAW=" in line:
                self.raw = int(float(line.split("RAW=")[1].split("|")[0].strip()))

            if "FILT=" in line:
                self.filtered = float(line.split("FILT=")[1].split("|")[0].strip())

            if "INT=" in line:
                self.integrated = float(line.split("INT=")[1].split("|")[0].strip())

            if "TH=" in line:
                self.threshold = float(line.split("TH=")[1].split("|")[0].strip())

            if "BPM_SMOOTH=" in line:
                self.bpm = float(line.split("BPM_SMOOTH=")[1].split("|")[0].strip())
            elif "BPM=" in line:
                self.bpm = float(line.split("BPM=")[1].split("|")[0].strip())

            if "STATUS=" in line:
                self.status = line.split("STATUS=")[1].strip()

            if self.status == "LEADS_OFF":
                self.force_leads_off(line)
                return

            if self.bpm > 0:
                self.rr = 60.0 / self.bpm
            else:
                self.rr = 0.0

            self.update_quality_from_status()

        except Exception:
            pass

    def parse_rpeak_log(self, line):
        try:
            if self.status == "LEADS_OFF":
                return

            self.stream_mode = "LOG"

            if "RR=" in line:
                self.rr = float(line.split("RR=")[1].split("|")[0].strip())

            if "BPM_SMOOTH=" in line:
                self.bpm = float(line.split("BPM_SMOOTH=")[1].split("|")[0].strip())
            elif "BPM=" in line:
                self.bpm = float(line.split("BPM=")[1].split("|")[0].strip())

            if "STATUS=" in line:
                self.status = line.split("STATUS=")[1].strip()

            self.update_quality_from_status()

        except Exception:
            pass

    def update_quality_from_status(self):
        if self.status == "LEADS_OFF":
            self.signal_quality = "BAD"
        elif self.bpm <= 0:
            self.signal_quality = "WAITING"
        elif abs(self.filtered) > 5000:
            self.signal_quality = "BAD"
        elif abs(self.filtered) > 2500:
            self.signal_quality = "ACCEPTABLE"
        else:
            self.signal_quality = "GOOD"

    def update_labels(self):
        self.bpm_label.config(text=f"BPM: {self.bpm:.1f}")
        self.rr_label.config(text=f"RR: {self.rr:.3f} s")
        self.status_label.config(text=f"STATUS: {self.status}")
        self.quality_label.config(text=f"Signal quality: {self.signal_quality}")
        self.stream_label.config(text=f"STREAM: {self.stream_mode}")
        self.tech_label.config(
            text=f"RAW: {self.raw} | FILT: {self.filtered:.1f} | INT: {self.integrated:.1f} | TH: {self.threshold:.1f}"
        )

        self.root.after(200, self.update_labels)

    def log(self, text):
        now = time.strftime("%H:%M:%S")
        self.terminal.insert("end", f"[{now}] {text}\n")
        self.terminal.see("end")


def main():
    root = tk.Tk()
    DoctorECGTerminal(root)
    root.mainloop()


if __name__ == "__main__":
    main()