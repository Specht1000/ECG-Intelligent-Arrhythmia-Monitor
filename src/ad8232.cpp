#include "ad8232.h"
#include "main.h"

void ad8232_init(void)
{
    pinMode(AD8232_LO_PLUS, INPUT);
    pinMode(AD8232_LO_MINUS, INPUT);

    analogReadResolution(12);
    analogSetPinAttenuation(AD8232_OUTPUT_PIN, ADC_11db);

    LOG("AD8232", "Driver inicializado");
    LOG("AD8232", "OUTPUT=GPIO%d LO+=GPIO%d LO-=GPIO%d",
        AD8232_OUTPUT_PIN,
        AD8232_LO_PLUS,
        AD8232_LO_MINUS);
}

int ad8232_read_raw(void)
{
    return analogRead(AD8232_OUTPUT_PIN);
}

int ad8232_get_lo_plus(void)
{
    return digitalRead(AD8232_LO_PLUS);
}

int ad8232_get_lo_minus(void)
{
    return digitalRead(AD8232_LO_MINUS);
}

bool ad8232_leads_off(void)
{
    return ad8232_get_lo_plus() == HIGH || ad8232_get_lo_minus() == HIGH;
}