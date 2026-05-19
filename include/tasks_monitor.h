#ifndef TASKS_MONITOR_H
#define TASKS_MONITOR_H

#include <Arduino.h>
#include <stdint.h>

typedef enum {
    TASK_MAIN_LOOP = 0,
    TASK_AD8232_READ,
    TASK_ECG_PROCESSING,
    MONITOR_COUNT
} TASKS_TIMER;

typedef struct {
    TickType_t startTime;
    TickType_t endTime;
    TickType_t executionTime;
    uint32_t executionCount;
} TaskExecutionTime;

extern TaskExecutionTime taskTimes[MONITOR_COUNT];

void taskMonitorTasks(void *pvParameters);
void startTaskTimer(TASKS_TIMER task);
void endTaskTimer(TASKS_TIMER task);

#endif