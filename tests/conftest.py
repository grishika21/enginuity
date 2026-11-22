"""Tests run without Redis or an LLM: storage goes to a temp folder and fakes stand in."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["STORAGE_DIR"] = tempfile.mkdtemp(prefix="enginuity-test-")
os.environ["JWT_SECRET"] = "test-secret"
os.environ.setdefault("DATA_DIR", str(ROOT / "data"))


class FakeRedis:
    """The handful of Redis commands the registry and file store use."""

    def __init__(self):
        self.kv, self.hashes, self.sets = {}, {}, {}

    def set(self, key, value, ex=None):
        self.kv[key] = str(value)

    def get(self, key):
        return self.kv.get(key)

    def exists(self, key):
        return int(key in self.kv)

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.hashes.pop(k, None)
            self.sets.pop(k, None)

    def hset(self, key, field=None, value=None, mapping=None):
        h = self.hashes.setdefault(key, {})
        if mapping:
            h.update({k: str(v) for k, v in mapping.items()})
        if field is not None:
            h[field] = value

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hdel(self, key, field):
        self.hashes.get(key, {}).pop(field, None)

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def sismember(self, key, value):
        return value in self.sets.get(key, set())

    def publish(self, channel, message):
        return 0
