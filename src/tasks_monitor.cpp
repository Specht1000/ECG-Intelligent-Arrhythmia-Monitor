#include "tasks_monitor.h"
#include "main.h"
#include "esp_system.h"

TaskExecutionTime taskTimes[MONITOR_COUNT] = {};

void taskMonitorTasks(void *pvParameters)
{
    (void)pvParameters;

    vTaskDelay(pdMS_TO_TICKS(5000));

    while (1) {
        LOG("MONITOR", "Task\tExecTime(ms)\tCount");
        LOG("MONITOR", "-------------------------------");

        for (int i = 0; i < MONITOR_COUNT; i++) {
            if (taskTimes[i].executionCount == 0) continue;

            uint32_t ms = taskTimes[i].executionTime * portTICK_PERIOD_MS;

            LOG("MONITOR", "%d\t%" PRIu32 "\t\t%" PRIu32,
                i,
                ms,
                taskTimes[i].executionCount);
        }

        vTaskDelay(pdMS_TO_TICKS(30000));
    }
}

void startTaskTimer(TASKS_TIMER task)
{
    taskTimes[task].startTime = xTaskGetTickCount();
}

void endTaskTimer(TASKS_TIMER task)
{
    taskTimes[task].endTime = xTaskGetTickCount();
    taskTimes[task].executionTime = taskTimes[task].endTime - taskTimes[task].startTime;
    taskTimes[task].executionCount++;
}