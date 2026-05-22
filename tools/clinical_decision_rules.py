from collections import Counter


def clinical_decision(
    beat_labels,
    bpm,
    signal_quality,
    min_beats=10,
    ventricular_alert_ratio=0.15,
    supraventricular_alert_ratio=0.25,
    tachycardia_bpm=120,
    bradycardia_bpm=50,
):
    """
    Regras clínicas simples para demonstração do sistema.

    beat_labels:
        lista com labels: normal, ventricular, supraventricular, low_confidence

    bpm:
        BPM estimado

    signal_quality:
        good, acceptable ou bad
    """

    if signal_quality == "bad":
        return {
            "decision": "signal_bad",
            "message": "Sinal ruim: reposicionar eletrodos",
            "severity": "warning",
        }

    valid_labels = [
        x for x in beat_labels
        if x not in ["low_confidence", "unknown", None]
    ]

    if len(valid_labels) < min_beats:
        return {
            "decision": "insufficient_data",
            "message": "Dados insuficientes para decisão",
            "severity": "info",
        }

    counts = Counter(valid_labels)
    total = len(valid_labels)

    ventricular_ratio = counts.get("ventricular", 0) / total
    supraventricular_ratio = counts.get("supraventricular", 0) / total

    if bpm >= tachycardia_bpm:
        return {
            "decision": "tachycardia",
            "message": f"Alerta: possível taquicardia ({bpm:.1f} BPM)",
            "severity": "alert",
        }

    if bpm <= bradycardia_bpm and bpm > 0:
        return {
            "decision": "bradycardia",
            "message": f"Alerta: possível bradicardia ({bpm:.1f} BPM)",
            "severity": "alert",
        }

    if ventricular_ratio >= ventricular_alert_ratio:
        return {
            "decision": "ventricular_alert",
            "message": (
                f"Alerta: batimentos ventriculares detectados "
                f"({ventricular_ratio * 100:.1f}%)"
            ),
            "severity": "alert",
        }

    if supraventricular_ratio >= supraventricular_alert_ratio:
        return {
            "decision": "supraventricular_alert",
            "message": (
                f"Atenção: batimentos supraventriculares frequentes "
                f"({supraventricular_ratio * 100:.1f}%)"
            ),
            "severity": "warning",
        }

    return {
        "decision": "normal",
        "message": f"Ritmo predominantemente normal ({bpm:.1f} BPM)",
        "severity": "normal",
    }


if __name__ == "__main__":
    tests = [
        {
            "beat_labels": ["normal"] * 20,
            "bpm": 75,
            "signal_quality": "good",
        },
        {
            "beat_labels": ["normal"] * 15 + ["ventricular"] * 5,
            "bpm": 82,
            "signal_quality": "good",
        },
        {
            "beat_labels": ["normal"] * 20,
            "bpm": 135,
            "signal_quality": "good",
        },
        {
            "beat_labels": ["normal"] * 20,
            "bpm": 45,
            "signal_quality": "acceptable",
        },
        {
            "beat_labels": ["normal"] * 20,
            "bpm": 80,
            "signal_quality": "bad",
        },
    ]

    for test in tests:
        result = clinical_decision(**test)
        print(result)