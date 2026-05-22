#include "ads1115_ecg.h"
#include "main.h"
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

static Adafruit_ADS1115 ads;

bool ads1115_ecg_init(void)
{
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

    if (!ads.begin(ADS1115_ADDR)) {
        LOG("ADS1115", "Erro ao inicializar ADS1115");
        return false;
    }

    ads.setGain(GAIN_ONE); 
    ads.setDataRate(RATE_ADS1115_860SPS);

    LOG("ADS1115", "Inicializado em 0x%02X", ADS1115_ADDR);
    LOG("ADS1115", "SDA=GPIO%d SCL=GPIO%d", I2C_SDA_PIN, I2C_SCL_PIN);

    return true;
}

int16_t ads1115_ecg_read_raw(void)
{
    return ads.readADC_SingleEnded(0);
}

float ads1115_ecg_read_voltage(void)
{
    int16_t raw = ads1115_ecg_read_raw();

    /*
     * GAIN_ONE = +/-4.096V
     * 1 bit = 0.125 mV
     */
    return raw * 0.000125f;
}