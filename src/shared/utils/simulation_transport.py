"""Coordinator-owned receive times for queued virtual clients."""
import time


class TimedClientProxy:
    def __init__(self, proxy, received):
        self._proxy, self._received = proxy, received

    def __getattr__(self, name):
        return getattr(self._proxy, name)

    def fit(self, *args, **kwargs):
        result = self._proxy.fit(*args, **kwargs)
        # Never accept a client-supplied timestamp as the receive time.
        self._received[self.cid] = time.time()
        return result
