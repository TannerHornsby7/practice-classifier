"""
Consumer — reads from Redis Stream using a consumer group, calls the
classifier service, and logs each prediction.

Consumer groups mean multiple consumer instances can share the stream —
each message is delivered to exactly one consumer. Scale by adding more
consumer replicas to docker-compose.
"""

import os
import time
import json
import redis
import requests

REDIS_URL      = os.environ.get("REDIS_URL", "redis://localhost:6379")
CLASSIFIER_URL = os.environ.get("CLASSIFIER_URL", "http://localhost:8000")
STREAM_NAME    = os.environ.get("STREAM_NAME", "appointments")
GROUP          = os.environ.get("CONSUMER_GROUP", "classifiers")
CONSUMER       = os.environ.get("CONSUMER_NAME", "consumer-1")

def _to_number(v: str) -> float:
    """Convert a Redis string value to float.
    Handles regular numbers, and boolean strings from one-hot encoded columns.
    """
    if v == "True":
        return 1.0
    if v == "False":
        return 0.0
    return float(v)


r = redis.from_url(REDIS_URL, decode_responses=True)

# create the consumer group (idempotent — fails silently if already exists)
try:
    r.xgroup_create(STREAM_NAME, GROUP, id="0", mkstream=True)
    print(f"Consumer group '{GROUP}' created on stream '{STREAM_NAME}'")
except redis.exceptions.ResponseError as e:
    if "BUSYGROUP" in str(e):
        print(f"Consumer group '{GROUP}' already exists — joining")
    else:
        raise

print(f"Consumer '{CONSUMER}' listening on '{STREAM_NAME}'...")

correct = 0
total   = 0

while True:
    # XREADGROUP: block up to 2s waiting for new messages
    results = r.xreadgroup(GROUP, CONSUMER, {STREAM_NAME: ">"}, count=10, block=2000)

    if not results:
        continue

    _, messages = results[0]

    for msg_id, fields in messages:
        true_label = fields.get("no_show", "unknown")

        try:
            resp = requests.post(
                f"{CLASSIFIER_URL}/predict",
                json={k: _to_number(v) for k, v in fields.items()},
                timeout=5,
            )
            resp.raise_for_status()
            result = resp.json()

            prediction = result["prediction"]
            p_no_show  = result["p_no_show"]

            # simple accuracy tracking
            total += 1
            predicted_label = "1" if prediction == "no_show" else "0"
            if predicted_label == str(int(float(true_label))):
                correct += 1

            accuracy = correct / total if total else 0
            print(
                f"  msg={msg_id} | true={'no_show' if true_label=='1' else 'show':7s} "
                f"| pred={prediction:7s} | p_no_show={p_no_show:.3f} "
                f"| running_acc={accuracy:.1%} ({correct}/{total})"
            )

        except Exception as e:
            print(f"  msg={msg_id} ERROR: {e}")

        # acknowledge: message won't be redelivered
        r.xack(STREAM_NAME, GROUP, msg_id)
