#include "ecg_processing.h"
#include "main.h"

static float prev_sample = 0.0f;
static float derivative_prev = 0.0f;

/*
 * Suavização extra antes da derivada.
 * Ajuda a reduzir ruído rápido do AD8232/ADS1115.
 */
static float lowpass_prev = 0.0f;

static float mwi_buffer[ECG_MWI_SIZE];
static int mwi_index = 0;
static float mwi_sum = 0.0f;

static float spki = 0.0f;
static float npki = 0.0f;
static float threshold_i = 0.0f;

static unsigned long last_peak_ms = 0;

static float rr_buffer[ECG_RR_BUFFER_SIZE];
static int rr_index = 0;
static int rr_count = 0;

static float bpm_current = 0.0f;

void ecg_processing_init(void)
{
    prev_sample = 0.0f;
    derivative_prev = 0.0f;
    lowpass_prev = 0.0f;

    for (int i = 0; i < ECG_MWI_SIZE; i++) {
        mwi_buffer[i] = 0.0f;
    }

    for (int i = 0; i < ECG_RR_BUFFER_SIZE; i++) {
        rr_buffer[i] = 0.0f;
    }

    mwi_index = 0;
    mwi_sum = 0.0f;

    spki = 0.0f;
    npki = 0.0f;
    threshold_i = 80.0f;

    last_peak_ms = 0;

    rr_index = 0;
    rr_count = 0;
    bpm_current = 0.0f;

    LOG("ECG", "Pan-Tompkins adaptativo inicializado");
}

static float update_rr_and_bpm(unsigned long now_ms)
{
    if (last_peak_ms == 0) {
        last_peak_ms = now_ms;
        return 0.0f;
    }

    float rr_sec = (now_ms - last_peak_ms) / 1000.0f;
    last_peak_ms = now_ms;

    if (rr_sec < 0.45f || rr_sec > 1.50f) {
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

ecg_output_t ecg_processing_update(int sample)
{
    ecg_output_t out = {};

    out.raw = sample;
    out.filtered = (float)sample;
    out.r_peak_detected = false;
    out.bpm = bpm_current;
    out.rr_sec = 0.0f;

    float x = (float)sample;

    /*
     * 0) Low-pass simples antes da derivada.
     * Quanto maior o primeiro coeficiente, mais suave fica.
     */
    x = 0.85f * lowpass_prev + 0.15f * x;
    lowpass_prev = x;

    out.filtered = x;

    /*
     * 1) Derivada
     */
    float derivative = x - prev_sample;
    prev_sample = x;

    /*
     * 2) Suavização da derivada.
     * Mais suave para reduzir falsos picos por ruído rápido.
     */
    derivative = 0.92f * derivative_prev + 0.08f * derivative;
    derivative_prev = derivative;

    /*
     * 3) Quadrado
     */
    float squared = derivative * derivative;

    /*
     * 4) Moving Window Integration
     */
    mwi_sum -= mwi_buffer[mwi_index];
    mwi_buffer[mwi_index] = squared;
    mwi_sum += squared;

    mwi_index = (mwi_index + 1) % ECG_MWI_SIZE;

    float integrated = mwi_sum / ECG_MWI_SIZE;
    out.integrated = integrated;

    /*
     * 5) Threshold adaptativo
     */
    threshold_i = npki + 0.25f * (spki - npki);
    out.threshold = threshold_i;

    unsigned long now_ms = millis();

    bool refractory_ok = (now_ms - last_peak_ms) > ECG_REFRACTORY_MS;

    bool is_peak =
        integrated > (threshold_i * 1.25f) &&
        derivative > 5.0f &&
        refractory_ok;

    if (is_peak) {
        spki = 0.125f * integrated + 0.875f * spki;

        out.r_peak_detected = true;
        out.rr_sec = update_rr_and_bpm(now_ms);
        out.bpm = bpm_current;
    } else {
        npki = 0.125f * integrated + 0.875f * npki;
    }

    return out;
}