#ifndef ECG_PROCESSING_H
#define ECG_PROCESSING_H

#include <Arduino.h>
#include <stdbool.h>

#define ECG_FS_HZ              100
#define ECG_MWI_SIZE           15
#define ECG_RR_BUFFER_SIZE     8
#define ECG_REFRACTORY_MS      300

typedef struct {
    bool r_peak_detected;
    float bpm;
    float rr_sec;
    int raw;
    float filtered;
    float integrated;
} ecg_output_t;

void ecg_processing_init(void);
ecg_output_t ecg_processing_update(int raw_sample);

#endif