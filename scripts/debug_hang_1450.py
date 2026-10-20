import sys
import json
sys.path.append('scripts')
from batch_test_engine import BatchTestEngine

with open('manifests/questions-index.json', encoding='utf-8') as f:
    idx = json.load(f)

engine = BatchTestEngine('.')
for i, q in enumerate(idx['questions'][1450:1500]):
    print(f"Checking {i+1451}: {q['slug']}", flush=True)
    cnt, status = engine.process_question(q)
    print(f" -> {cnt} ({status})", flush=True)
engine.close()
