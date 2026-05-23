#include "main.h"

#include "ads1115_ecg.h"
#include "ecg_processing.h"
#include "ecg_filter.h"
#include "sh1106_oled.h"

#define ECG_TASK_STACK_SIZE 8192
#define ECG_TASK_PRIORITY   2

#define ECG_LO_PLUS_PIN     4
#define ECG_LO_MINUS_PIN    5

#define LEADS_OFF_COUNT_LIMIT 8
#define BPM_HISTORY_SIZE 8

typedef enum {
    STREAM_MODE_LOG = 0,
    STREAM_MODE_DATA = 1
} stream_mode_t;

typedef enum {
    DEMO_OFF = 0,
    DEMO_NORMAL,
    DEMO_BRADY,
    DEMO_TACHY
} demo_mode_t;

static stream_mode_t stream_mode = STREAM_MODE_LOG;
static demo_mode_t demo_mode = DEMO_OFF;

static sh1106_t oled;
static bool oled_ok = false;

static unsigned long last_debug_ms = 0;
static unsigned long last_oled_ms = 0;
static unsigned long last_demo_ms = 0;

static float g_bpm = 0.0f;
static float g_bpm_smoothed = 0.0f;
static bool g_bpm_initialized = false;

static char g_status[32] = "STARTING";

static int leads_off_counter = 0;

static float bpm_history[BPM_HISTORY_SIZE];
static int bpm_hist_index = 0;
static int bpm_hist_count = 0;

static bool ecg_leads_off_raw()
{
    bool lo_plus = digitalRead(ECG_LO_PLUS_PIN) == HIGH;
    bool lo_minus = digitalRead(ECG_LO_MINUS_PIN) == HIGH;

    return lo_plus || lo_minus;
}

static bool ecg_leads_off_debounced()
{
    bool raw = ecg_leads_off_raw();

    if (raw) {
        if (leads_off_counter < LEADS_OFF_COUNT_LIMIT) {
            leads_off_counter++;
        }
    } else {
        if (leads_off_counter > 0) {
            leads_off_counter--;
        }
    }

    return leads_off_counter >= LEADS_OFF_COUNT_LIMIT;
}

static float median_bpm()
{
    if (bpm_hist_count == 0) {
        return 0.0f;
    }

    float sorted[BPM_HISTORY_SIZE];

    for (int i = 0; i < bpm_hist_count; i++) {
        sorted[i] = bpm_history[i];
    }

    for (int i = 0; i < bpm_hist_count - 1; i++) {
        for (int j = i + 1; j < bpm_hist_count; j++) {
            if (sorted[j] < sorted[i]) {
                float tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }
        }
    }

    return sorted[bpm_hist_count / 2];
}

static float smooth_bpm(float bpm)
{
    if (bpm < 35.0f || bpm > 180.0f) {
        return g_bpm_smoothed;
    }

    bpm_history[bpm_hist_index] = bpm;
    bpm_hist_index = (bpm_hist_index + 1) % BPM_HISTORY_SIZE;

    if (bpm_hist_count < BPM_HISTORY_SIZE) {
        bpm_hist_count++;
    }

    float bpm_med = median_bpm();

    if (!g_bpm_initialized) {
        g_bpm_smoothed = bpm_med;
        g_bpm_initialized = true;
        return g_bpm_smoothed;
    }

    float max_step = 1.5f;
    float diff = bpm_med - g_bpm_smoothed;

    if (diff > max_step) {
        diff = max_step;
    }

    if (diff < -max_step) {
        diff = -max_step;
    }

    g_bpm_smoothed = g_bpm_smoothed + 0.15f * diff;

    return g_bpm_smoothed;
}

static void reset_bpm_filter()
{
    g_bpm = 0.0f;
    g_bpm_smoothed = 0.0f;
    g_bpm_initialized = false;

    bpm_hist_index = 0;
    bpm_hist_count = 0;

    for (int i = 0; i < BPM_HISTORY_SIZE; i++) {
        bpm_history[i] = 0.0f;
    }
}

static void update_status_from_bpm(float bpm)
{
    if (bpm <= 0.0f) {
        snprintf(g_status, sizeof(g_status), "WAITING");
    } else if (bpm < 50.0f) {
        snprintf(g_status, sizeof(g_status), "LOW_HR");
    } else if (bpm > 120.0f) {
        snprintf(g_status, sizeof(g_status), "HIGH_HR");
    } else {
        snprintf(g_status, sizeof(g_status), "NORMAL");
    }
}

static void handle_serial_commands()
{
    while (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        if (cmd == "MODE_AI" || cmd == "STREAM_ON" || cmd == "DATA") {
            stream_mode = STREAM_MODE_DATA;
            LOG("SERIAL", "Mode AI: DATA stream ON");
            return;
        }

        if (cmd == "MODE_DOCTOR" || cmd == "STREAM_OFF" || cmd == "LOG") {
            stream_mode = STREAM_MODE_LOG;
            LOG("SERIAL", "Mode Doctor: LOG stream ON");
            return;
        }

        if (cmd == "DEMO_TACHY") {
            demo_mode = DEMO_TACHY;
            reset_bpm_filter();
            LOG("DEMO", "Demo mode: TACHYCARDIA");
            return;
        }

        if (cmd == "DEMO_BRADY") {
            demo_mode = DEMO_BRADY;
            reset_bpm_filter();
            LOG("DEMO", "Demo mode: BRADYCARDIA");
            return;
        }

        if (cmd == "DEMO_NORMAL") {
            demo_mode = DEMO_NORMAL;
            reset_bpm_filter();
            LOG("DEMO", "Demo mode: NORMAL");
            return;
        }

        if (cmd == "DEMO_OFF") {
            demo_mode = DEMO_OFF;
            reset_bpm_filter();
            LOG("DEMO", "Demo mode OFF - real ECG restored");
            return;
        }
    }
}

static void updateOLED()
{
    if (!oled_ok) {
        return;
    }

    sh1106_clear(&oled);

    char line[64];

    sh1106_draw_text_line(&oled, 0, "PFE ECG MONITOR");

    snprintf(line, sizeof(line), "BPM: %.1f", g_bpm);
    sh1106_draw_text_line(&oled, 2, line);

    sh1106_draw_text_line(&oled, 4, "STATUS:");

    snprintf(line, sizeof(line), "%s", g_status);
    sh1106_draw_text_line(&oled, 5, line);

    if (demo_mode != DEMO_OFF) {
        sh1106_draw_text_line(&oled, 7, "DEMO MODE");
    }

    sh1106_refresh(&oled);
}

static void send_demo_sample()
{
    unsigned long now_ms = millis();
    unsigned long now_us = micros();

    if ((now_ms - last_demo_ms) < 120) {
        return;
    }

    last_demo_ms = now_ms;

    static int demo_index = 0;
    demo_index++;

    float bpm = 75.0f;
    float rr = 0.800f;
    const char *status = "NORMAL";

    if (demo_mode == DEMO_TACHY) {
        bpm = 135.0f;
        rr = 0.444f;
        status = "HIGH_HR";
    } else if (demo_mode == DEMO_BRADY) {
        bpm = 42.0f;
        rr = 1.428f;
        status = "LOW_HR";
    } else if (demo_mode == DEMO_NORMAL) {
        bpm = 75.0f;
        rr = 0.800f;
        status = "NORMAL";
    }

    g_bpm = bpm;
    snprintf(g_status, sizeof(g_status), "%s", status);

    float phase = (float)(demo_index % 20);

    int raw = 12000;
    float filt = 0.0f;

    if (phase == 0) {
        raw = 15000;
        filt = 1200.0f;
    } else if (phase == 1) {
        raw = 21000;
        filt = 4200.0f;
    } else if (phase == 2) {
        raw = 14500;
        filt = 900.0f;
    } else if (phase == 3) {
        raw = 11000;
        filt = -600.0f;
    } else {
        raw = 12000 + ((demo_index % 7) * 80);
        filt = -120.0f + ((demo_index % 5) * 40.0f);
    }

    float integrated = (phase <= 2) ? 95000.0f : 18000.0f;
    float threshold = 40000.0f;
    int rpeak = (phase == 1) ? 1 : 0;

    if (stream_mode == STREAM_MODE_DATA) {
        Serial.printf(
            "DATA,%lu,%d,%.2f,%.2f,%d,%.1f,%.2f,%s\n",
            now_us,
            raw,
            filt,
            integrated,
            rpeak,
            g_bpm,
            threshold,
            g_status
        );
    }

    if ((now_ms - last_debug_ms) >= 1000) {
        last_debug_ms = now_ms;

        LOG(
            "ECG",
            "DEMO | RAW=%d | FILT=%.1f | INT=%.1f | TH=%.1f | BPM=%.1f | STATUS=%s",
            raw,
            filt,
            integrated,
            threshold,
            g_bpm,
            g_status
        );
    }

    if (rpeak) {
        LOG(
            "RPEAK",
            "DEMO | RR=%.3f | BPM_INST=%.1f | BPM_SMOOTH=%.1f | STATUS=%s",
            rr,
            bpm,
            bpm,
            status
        );
    }

    if ((now_ms - last_oled_ms) >= 500) {
        last_oled_ms = now_ms;
        updateOLED();
    }
}

static void ecgTask(void *pvParameters)
{
    unsigned long last_sample_us = micros();

    while (true) {
        handle_serial_commands();

        if (demo_mode != DEMO_OFF) {
            send_demo_sample();
            vTaskDelay(1);
            continue;
        }

        unsigned long now_us = micros();

        if ((now_us - last_sample_us) >= SAMPLE_INTERVAL_US) {
            last_sample_us += SAMPLE_INTERVAL_US;

            startTaskTimer(TASK_AD8232_READ);
            int16_t raw = ads1115_ecg_read_raw();
            endTaskTimer(TASK_AD8232_READ);

            bool leads_off = ecg_leads_off_debounced();

            if (leads_off) {
                reset_bpm_filter();
                snprintf(g_status, sizeof(g_status), "LEADS_OFF");

                if (stream_mode == STREAM_MODE_DATA) {
                    Serial.printf(
                        "DATA,%lu,%d,0.00,0.00,0,0.0,0.00,%s\n",
                        now_us,
                        raw,
                        g_status
                    );
                }

                unsigned long now_ms = millis();

                if ((now_ms - last_debug_ms) >= 1000) {
                    last_debug_ms = now_ms;

                    LOG(
                        "ECG",
                        "LEADS_OFF | RAW=%d | BPM=0.0",
                        raw
                    );
                }

                if ((now_ms - last_oled_ms) >= 500) {
                    last_oled_ms = now_ms;
                    updateOLED();
                }

                vTaskDelay(1);
                continue;
            }

            float ecg_filtered = (float)raw;

            ecg_filtered = ecg_highpass(ecg_filtered);
            ecg_filtered = ecg_lowpass(ecg_filtered);
            ecg_filtered = ecg_moving_average(ecg_filtered);

            startTaskTimer(TASK_ECG_PROCESSING);
            ecg_output_t ecg = ecg_processing_update((int)ecg_filtered);
            endTaskTimer(TASK_ECG_PROCESSING);

            if (ecg.bpm > 0.0f) {
                g_bpm = smooth_bpm(ecg.bpm);
            }

            update_status_from_bpm(g_bpm);

            unsigned long now_ms = millis();

            if (stream_mode == STREAM_MODE_DATA) {
                Serial.printf(
                    "DATA,%lu,%d,%.2f,%.2f,%d,%.1f,%.2f,%s\n",
                    now_us,
                    raw,
                    ecg_filtered,
                    ecg.integrated,
                    ecg.r_peak_detected ? 1 : 0,
                    g_bpm,
                    ecg.threshold,
                    g_status
                );
            }

            if ((now_ms - last_debug_ms) >= 2000) {
                last_debug_ms = now_ms;

                LOG(
                    "ECG",
                    "RAW=%d | FILT=%.1f | INT=%.1f | TH=%.1f | BPM=%.1f | STATUS=%s",
                    raw,
                    ecg_filtered,
                    ecg.integrated,
                    ecg.threshold,
                    g_bpm,
                    g_status
                );
            }

            if (ecg.r_peak_detected) {
                LOG(
                    "RPEAK",
                    "RR=%.3f | BPM_INST=%.1f | BPM_SMOOTH=%.1f | STATUS=%s",
                    ecg.rr_sec,
                    ecg.bpm,
                    g_bpm,
                    g_status
                );
            }

            if ((now_ms - last_oled_ms) >= 500) {
                last_oled_ms = now_ms;
                updateOLED();
            }
        }

        vTaskDelay(1);
    }
}

void setup()
{
    Serial.begin(BAUD_RATE);
    delay(1500);

    pinMode(ECG_LO_PLUS_PIN, INPUT);
    pinMode(ECG_LO_MINUS_PIN, INPUT);

    LOG("BOOT", "=================================");
    LOG("BOOT", "PFE ECG STARTING");
    LOG("BOOT", "ESP32-S3 + ADS1115 + AD8232");
    LOG("BOOT", "MODE_DOCTOR = LOG");
    LOG("BOOT", "MODE_AI = DATA");
    LOG("BOOT", "DEMO_TACHY / DEMO_BRADY / DEMO_NORMAL / DEMO_OFF");
    LOG("BOOT", "Default mode = LOG");
    LOG("BOOT", "=================================");

    if (!ads1115_ecg_init()) {
        LOG("BOOT", "ADS1115 failed");

        while (true) {
            delay(1000);
        }
    }

    oled_ok = sh1106_init(&oled, 0, 0x3C);

    if (oled_ok) {
        LOG("BOOT", "OLED initialized");

        sh1106_clear(&oled);
        sh1106_draw_text_line(&oled, 0, "PFE ECG");
        sh1106_draw_text_line(&oled, 2, "Starting...");
        sh1106_refresh(&oled);
    } else {
        LOG("BOOT", "OLED not initialized");
    }

    ecg_filter_init();
    ecg_processing_init();

    xTaskCreate(
        taskMonitorTasks,
        "taskMonitor",
        4096,
        NULL,
        1,
        NULL
    );

    xTaskCreate(
        ecgTask,
        "ecgTask",
        ECG_TASK_STACK_SIZE,
        NULL,
        ECG_TASK_PRIORITY,
        NULL
    );

    LOG("BOOT", "System initialized");
    LOG("BOOT", "Sampling rate = %d Hz", SAMPLE_RATE_HZ);
}

void loop()
{
    vTaskDelay(1000 / portTICK_PERIOD_MS);
}