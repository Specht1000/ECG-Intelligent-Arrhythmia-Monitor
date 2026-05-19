#include "main.h"
#include "ad8232.h"
#include "ecg_processing.h"

static unsigned long last_sample_us = 0;

void setup()
{
    Serial.begin(BAUD_RATE);
    delay(1000);

    LOG("BOOT", "PFE ECG starting...");
    LOG("BOOT", "Framework Arduino");

    ad8232_init();
    ecg_processing_init();

    xTaskCreate(
        taskMonitorTasks,
        "taskMonitor",
        4096,
        NULL,
        1,
        NULL
    );

    last_sample_us = micros();

    LOG("BOOT", "Sistema inicializado");
    LOG("BOOT", "Formato: timestamp_us,raw,integrated,r_peak,bpm");
}

void loop()
{
    startTaskTimer(TASK_MAIN_LOOP);

    unsigned long now_us = micros();

    if ((now_us - last_sample_us) >= SAMPLE_INTERVAL_US) {
        last_sample_us += SAMPLE_INTERVAL_US;

        startTaskTimer(TASK_AD8232_READ);
        int raw = ad8232_read_raw();
        endTaskTimer(TASK_AD8232_READ);

        startTaskTimer(TASK_ECG_PROCESSING);
        ecg_output_t ecg = ecg_processing_update(raw);
        endTaskTimer(TASK_ECG_PROCESSING);

        printf(
            "%lu,%d,%.2f,%d,%.1f\n",
            now_us,
            raw,
            ecg.integrated,
            ecg.r_peak_detected ? 1 : 0,
            ecg.bpm
        );

        if (ecg.r_peak_detected) {
            LOG("ECG", "R_PEAK RR=%.3f BPM=%.1f", ecg.rr_sec, ecg.bpm);
        }
    }

    endTaskTimer(TASK_MAIN_LOOP);
}