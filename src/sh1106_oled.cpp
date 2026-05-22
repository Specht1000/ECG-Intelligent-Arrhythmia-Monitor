#include "sh1106_oled.h"

static U8G2_SH1106_128X64_NONAME_F_HW_I2C g_display(
    U8G2_R0,
    U8X8_PIN_NONE
);

bool sh1106_init(
    sh1106_t *dev,
    int i2c_port,
    uint8_t addr
)
{
    (void)i2c_port;
    (void)addr;

    Wire.begin(8, 9);

    g_display.begin();

    g_display.clearBuffer();

    g_display.setFont(
        u8g2_font_6x12_tf
    );

    g_display.sendBuffer();

    dev->display = &g_display;

    return true;
}

void sh1106_clear(
    sh1106_t *dev
)
{
    dev->display->clearBuffer();
}

void sh1106_draw_text_line(
    sh1106_t *dev,
    uint8_t line,
    const char *text
)
{
    int y = (line * 10) + 10;

    dev->display->drawStr(
        0,
        y,
        text
    );
}

void sh1106_refresh(
    sh1106_t *dev
)
{
    dev->display->sendBuffer();
}