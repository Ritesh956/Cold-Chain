"""NATS JetStream names shared by every service, so producers and consumers cannot diverge."""

SENSOR_STREAM = "SENSORS"
SENSOR_SUBJECTS = ["sensor.>"]
SENSOR_MAX_AGE_S = 7 * 24 * 3600

ALERT_STREAM = "ALERTS"
ALERT_SUBJECTS = ["alerts.>"]
ALERT_MAX_AGE_S = 30 * 24 * 3600


def alert_subject(kind: str, zone: str) -> str:
    return f"alerts.{kind}.{zone.lower()}"
