"""
Producer — replays stream.csv into a Redis Stream row by row.
Simulates appointments arriving in real time.
"""

import os
import time
import redis
import pandas as pd

REDIS_URL   = os.environ.get("REDIS_URL", "redis://localhost:6379")
STREAM_NAME = os.environ.get("STREAM_NAME", "appointments")
DELAY       = float(os.environ.get("DELAY_SECONDS", "0.5"))

r = redis.from_url(REDIS_URL, decode_responses=True)

df = pd.read_csv("/app/stream.csv")
print(f"Producer: loaded {len(df)} records — publishing to '{STREAM_NAME}' every {DELAY}s")

for i, row in df.iterrows():
    # Redis Streams require all values to be strings
    record = {k: str(v) for k, v in row.items()}
    msg_id = r.xadd(STREAM_NAME, record)
    print(f"  [{i+1}/{len(df)}] published msg_id={msg_id}")
    time.sleep(DELAY)

print("Producer: all records published.")
