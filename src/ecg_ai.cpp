#include "ecg_ai.h"
#include "main.h"
#include "heartbeat_model.h"

bool ecg_ai_init(void)
{
    LOG("AI", "Mock AI initialized");
    LOG("AI", "Model header loaded, size=%u bytes", heartbeat_model_len);
    return true;
}

ecg_ai_result_t ecg_ai_predict(const float *beat, int len, float *confidence)
{
    float max_abs = 0.0f;

    for (int i = 0; i < len; i++) {
        float v = beat[i];

        if (v < 0.0f) {
            v = -v;
        }

        if (v > max_abs) {
            max_abs = v;
        }
    }

    if (confidence) {
        *confidence = 0.80f;
    }

    if (max_abs > 2000.0f) {
        return ECG_AI_VENTRICULAR;
    }

    return ECG_AI_NORMAL;
}

const char *ecg_ai_result_to_string(ecg_ai_result_t result)
{
    switch (result) {
        case ECG_AI_NORMAL:
            return "normal";

        case ECG_AI_SUPRAVENTRICULAR:
            return "supraventricular";

        case ECG_AI_VENTRICULAR:
            return "ventricular";

        default:
            return "unknown";
    }
}