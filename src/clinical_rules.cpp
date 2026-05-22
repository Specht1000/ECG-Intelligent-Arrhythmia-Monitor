#include "clinical_rules.h"

clinical_decision_t clinical_rules_evaluate(
    ecg_ai_result_t ai_result,
    float confidence,
    float bpm
)
{
    /*
     * Baixa confiança
     */
    if (confidence < 0.60f) {
        return CLINICAL_WARNING;
    }

    /*
     * BPM crítico
     */
    if (bpm < 45.0f || bpm > 130.0f) {
        return CLINICAL_ALERT;
    }

    /*
     * Arritmia ventricular
     */
    if (ai_result == ECG_AI_VENTRICULAR) {
        return CLINICAL_ALERT;
    }

    /*
     * Arritmia supraventricular
     */
    if (ai_result == ECG_AI_SUPRAVENTRICULAR) {
        return CLINICAL_WARNING;
    }

    return CLINICAL_NORMAL;
}

const char* clinical_decision_to_string(
    clinical_decision_t decision
)
{
    switch (decision)
    {
        case CLINICAL_NORMAL:
            return "NORMAL";

        case CLINICAL_WARNING:
            return "WARNING";

        case CLINICAL_ALERT:
            return "ALERT";

        default:
            return "UNKNOWN";
    }
}