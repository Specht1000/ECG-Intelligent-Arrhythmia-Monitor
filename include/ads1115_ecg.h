#ifndef ADS1115_ECG_H
#define ADS1115_ECG_H

#include <Arduino.h>

#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9
#define ADS1115_ADDR 0x48

bool ads1115_ecg_init(void);
int16_t ads1115_ecg_read_raw(void);
float ads1115_ecg_read_voltage(void);

#endif