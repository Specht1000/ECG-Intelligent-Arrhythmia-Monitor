#ifndef AD8232_H
#define AD8232_H

#include <Arduino.h>
#include <stdbool.h>

#define AD8232_OUTPUT_PIN 1
#define AD8232_LO_PLUS    2
#define AD8232_LO_MINUS   3

void ad8232_init(void);
int ad8232_read_raw(void);
int ad8232_get_lo_plus(void);
int ad8232_get_lo_minus(void);
bool ad8232_leads_off(void);

#endif