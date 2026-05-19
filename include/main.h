#ifndef MAIN_H
#define MAIN_H

#include <Arduino.h>
#include <stdio.h>
#include <inttypes.h>
#include "tasks_monitor.h"

#define ENABLE_DEBUG

#ifdef ENABLE_DEBUG
    #define LOG(tag, fmt, ...) printf("[" tag "] " fmt "\n", ##__VA_ARGS__)
#else
    #define LOG(tag, fmt, ...) ((void)0)
#endif

#define ECG_PIN 1
#define LO_PLUS_PIN 2
#define LO_MINUS_PIN 3

#define SAMPLE_RATE_HZ 100
#define BAUD_RATE 115200
#define SAMPLE_INTERVAL_US (1000000UL / SAMPLE_RATE_HZ)

#endif