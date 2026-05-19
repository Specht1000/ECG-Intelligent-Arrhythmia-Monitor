#include "ecg_processing.h"
#include "main.h"

static float prev_raw = 0.0f;
static float dc_estimate = 0.0f;

static float derivative_prev = 0.0f;

static float mwi_buffer[ECG_MWI_SIZE];
static int mwi_index = 0;
static float mwi_sum = 0.0f;

static float threshold = 1000.0f;

static unsigned long last_peak_ms = 0;

static float rr_buffer[ECG_RR_BUFFER_SIZE];
static int rr_index = 0;
static int rr_count = 0;

static float bpm_current = 0.0f;

void ecg_processing_init(void)
{
    prev_raw = 0.0f;
    dc_estimate = 0.0f;
    derivative_prev = 0.0f;

    for (int i = 0; i < ECG_MWI_SIZE; i++) {
        mwi_buffer[i] = 0.0f;
    }

    for (int i = 0; i < ECG_RR_BUFFER_SIZE; i++) {
        rr_buffer[i] = 0.0f;
    }

    mwi_index = 0;
    mwi_sum = 0.0f;
    threshold = 1000.0f;
    last_peak_ms = 0;
    rr_index = 0;
    rr_count = 0;
    bpm_current = 0.0f;

    LOG("ECG", "Pan-Tompkins simplificado inicializado");
}

static float update_rr_and_bpm(unsigned long now_ms)
{
    if (last_peak_ms == 0) {
        last_peak_ms = now_ms;
        return 0.0f;
    }

    float rr_sec = (now_ms - last_peak_ms) / 1000.0f;
    last_peak_ms = now_ms;

    if (rr_sec < 0.3f || rr_sec > 2.0f) {
        return 0.0f;
    }

    rr_buffer[rr_index] = rr_sec;
    rr_index = (rr_index + 1) % ECG_RR_BUFFER_SIZE;

    if (rr_count < ECG_RR_BUFFER_SIZE) {
        rr_count++;
    }

    float rr_sum = 0.0f;

    for (int i = 0; i < rr_count; i++) {
        rr_sum += rr_buffer[i];
    }

    float rr_mean = rr_sum / rr_count;

    if (rr_mean > 0.0f) {
        bpm_current = 60.0f / rr_mean;
    }

    return rr_sec;
}

ecg_output_t ecg_processing_update(int raw_sample)
{
    ecg_output_t out = {};
    out.raw = raw_sample;
    out.r_peak_detected = false;
    out.bpm = bpm_current;
    out.rr_sec = 0.0f;

    float x = (float)raw_sample;

    /*
     * 1) Remoção lenta de baseline/DC
     */
    dc_estimate = 0.995f * dc_estimate + 0.005f * x;
    float centered = x - dc_estimate;

    /*
     * 2) Derivada simples
     */
    float derivative = centered - prev_raw;
    prev_raw = centered;

    /*
     * 3) Suavização leve da derivada
     */
    derivative = 0.7f * derivative_prev + 0.3f * derivative;
    derivative_prev = derivative;

    /*
     * 4) Quadrado
     */
    float squared = derivative * derivative;

    /*
     * 5) Moving Window Integration
     */
    mwi_sum -= mwi_buffer[mwi_index];
    mwi_buffer[mwi_index] = squared;
    mwi_sum += squared;

    mwi_index = (mwi_index + 1) % ECG_MWI_SIZE;

    float integrated = mwi_sum / ECG_MWI_SIZE;

    /*
     * 6) Threshold adaptativo simples
     */
    threshold = 0.995f * threshold + 0.005f * integrated;

    float dynamic_threshold = threshold * 3.0f;

    unsigned long now_ms = millis();

    if (integrated > dynamic_threshold &&
        (now_ms - last_peak_ms) > ECG_REFRACTORY_MS) {

        out.r_peak_detected = true;
        out.rr_sec = update_rr_and_bpm(now_ms);
        out.bpm = bpm_current;
    }

    out.filtered = centered;
    out.integrated = integrated;

    return out;
}