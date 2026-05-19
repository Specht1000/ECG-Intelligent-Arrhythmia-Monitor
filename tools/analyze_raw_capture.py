import pandas as pd
import matplotlib.pyplot as plt

FILE = "ecg_raw_capture.csv"

df = pd.read_csv(FILE)

valid = df[df["lo_status"] == 0].copy()

adc = valid["adc"]

print("Amostras:", len(adc))
print("Min:", adc.min())
print("Max:", adc.max())
print("Média:", adc.mean())
print("Amplitude:", adc.max() - adc.min())
print("Saturação baixa:", (adc <= 5).sum())
print("Saturação alta:", (adc >= 4090).sum())

plt.figure(figsize=(12, 5))
plt.plot(adc.values)
plt.title("ECG bruto capturado")
plt.xlabel("Amostras")
plt.ylabel("ADC")
plt.grid(True)
plt.show()