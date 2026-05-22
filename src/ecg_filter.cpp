#include "ecg_filter.h"

/*
 * Filtros ajustados para FS = 250 Hz
 */

#define HP_ALPHA 0.98f
#define LP_ALPHA 0.20f
#define MA_SIZE 3

static float hp_prev_y = 0.0f;
static float hp_prev_x = 0.0f;

static float lp_prev_y = 0.0f;

static float ma_buffer[MA_SIZE];
static int ma_index = 0;

/*
 * Notch 50 Hz biquad para FS=250 Hz
 * Coeficientes aproximados.
 */
static float notch_x1 = 0.0f;
static float notch_x2 = 0.0f;
static float notch_y1 = 0.0f;
static float notch_y2 = 0.0f;

static const float b0 = 0.9565f;
static const float b1 = -0.5913f;
static const float b2 = 0.9565f;
static const float a1 = -0.5913f;
static const float a2 = 0.9131f;

void ecg_filter_init(void)
{
    hp_prev_y = 0.0f;
    hp_prev_x = 0.0f;
    lp_prev_y = 0.0f;

    notch_x1 = 0.0f;
    notch_x2 = 0.0f;
    notch_y1 = 0.0f;
    notch_y2 = 0.0f;

    for (int i = 0; i < MA_SIZE; i++) {
        ma_buffer[i] = 0.0f;
    }

    ma_index = 0;
}

float ecg_notch_50hz(float x)
{
    float y = b0 * x + b1 * notch_x1 + b2 * notch_x2
              - a1 * notch_y1 - a2 * notch_y2;

    notch_x2 = notch_x1;
    notch_x1 = x;

    notch_y2 = notch_y1;
    notch_y1 = y;

    return y;
}

float ecg_highpass(float x)
{
    float y = HP_ALPHA * (hp_prev_y + x - hp_prev_x);

    hp_prev_x = x;
    hp_prev_y = y;

    return y;
}

float ecg_lowpass(float x)
{
    float y = lp_prev_y + LP_ALPHA * (x - lp_prev_y);

    lp_prev_y = y;

    return y;
}

float ecg_moving_average(float x)
{
    ma_buffer[ma_index] = x;
    ma_index = (ma_index + 1) % MA_SIZE;

    float sum = 0.0f;

    for (int i = 0; i < MA_SIZE; i++) {
        sum += ma_buffer[i];
    }

    return sum / MA_SIZE;
}