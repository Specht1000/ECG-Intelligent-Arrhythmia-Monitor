#ifndef SH1106_OLED_H
#define SH1106_OLED_H

#include <Arduino.h>
#include <Wire.h>
#include <U8g2lib.h>

typedef struct
{
    U8G2_SH1106_128X64_NONAME_F_HW_I2C *display;

} sh1106_t;

bool sh1106_init(
    sh1106_t *dev,
    int i2c_port,
    uint8_t addr
);

void sh1106_clear(
    sh1106_t *dev
);

void sh1106_draw_text_line(
    sh1106_t *dev,
    uint8_t line,
    const char *text
);

void sh1106_refresh(
    sh1106_t *dev
);

#endif