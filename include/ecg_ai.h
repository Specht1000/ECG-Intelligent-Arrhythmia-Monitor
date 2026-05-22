#ifndef ECG_AI_H
#define ECG_AI_H

#include <Arduino.h>

typedef enum {
    ECG_AI_NORMAL = 0,
    ECG_AI_SUPRAVENTRICULAR,
    ECG_AI_VENTRICULAR,
    ECG_AI_UNKNOWN
} ecg_ai_result_t;

bool ecg_ai_init(void);

ecg_ai_result_t ecg_ai_predict(
    const float *beat,
    int len,
    float *confidence
);

const char *ecg_ai_result_to_string(ecg_ai_result_t result);

#endif