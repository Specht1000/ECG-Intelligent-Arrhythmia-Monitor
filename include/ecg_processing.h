#ifndef ECG_PROCESSING_H
#define ECG_PROCESSING_H

#include <Arduino.h>
#include <stdbool.h>

#define ECG_FS_HZ              250
#define ECG_MWI_SIZE           15
#define ECG_RR_BUFFER_SIZE     8

/*
 * Antes estava 300/500 ms.
 * Agora usamos 650 ms para evitar detectar R e S como dois batimentos.
 */
#define ECG_REFRACTORY_MS      650

typedef struct {
    bool r_peak_detected;
    float bpm;
    float rr_sec;
    int raw;
    float filtered;
    float integrated;
    float threshold;
} ecg_output_t;

void ecg_processing_init(void);
ecg_output_t ecg_processing_update(int sample);

#endif