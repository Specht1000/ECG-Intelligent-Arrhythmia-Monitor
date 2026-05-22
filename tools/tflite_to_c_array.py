from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "models" / "exported" / "heartbeat_cnn.tflite"
OUTPUT_FILE = PROJECT_ROOT / "firmware" / "main" / "heartbeat_model.h"

data = INPUT_FILE.read_bytes()

with open(OUTPUT_FILE, "w") as f:

    f.write("#ifndef HEARTBEAT_MODEL_H\n")
    f.write("#define HEARTBEAT_MODEL_H\n\n")

    f.write("const unsigned char heartbeat_model[] = {\n")

    for i, b in enumerate(data):

        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{b:02x},")

        if i % 12 == 11:
            f.write("\n")
        else:
            f.write(" ")

    f.write("\n};\n\n")

    f.write(f"const unsigned int heartbeat_model_len = {len(data)};\n\n")

    f.write("#endif\n")

print("Header C gerado:")
print(OUTPUT_FILE)
print(f"Tamanho: {len(data)} bytes")