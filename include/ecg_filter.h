#ifndef ECG_FILTER_H
#define ECG_FILTER_H

void ecg_filter_init(void);

float ecg_notch_50hz(float x);
float ecg_highpass(float x);
float ecg_lowpass(float x);
float ecg_moving_average(float x);

#endif