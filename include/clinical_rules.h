#ifndef CLINICAL_RULES_H
#define CLINICAL_RULES_H

#include "ecg_ai.h"

typedef enum
{
    CLINICAL_NORMAL = 0,
    CLINICAL_WARNING,
    CLINICAL_ALERT

} clinical_decision_t;

clinical_decision_t clinical_rules_evaluate(
    ecg_ai_result_t ai_result,
    float confidence,
    float bpm
);

const char* clinical_decision_to_string(
    clinical_decision_t decision
);

#endif