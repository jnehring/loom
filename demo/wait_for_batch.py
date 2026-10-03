"""Block until the single pending OpenAI batch in $LOOM_HOME has finished (used by demo.tape)."""
import json, os, pathlib, time
from openai import OpenAI

meta = next(pathlib.Path(os.environ["LOOM_HOME"], "batches").glob("*.json"))
batch_id = json.loads(meta.read_text())["batch_id"]
client = OpenAI()
while client.batches.retrieve(batch_id).status not in ("completed", "failed", "expired", "cancelled"):
    time.sleep(15)
