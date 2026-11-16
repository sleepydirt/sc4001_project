"""Persistent cumulative reserved-GPU wall accounting, including interrupted work."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
from .common import STUDY, write

class BudgetStop(RuntimeError):
    pass

class GPULedger:
    def __init__(self, root=STUDY, limit=36000., margin=120.):
        self.path = Path(root)/'gpu_budget.json'
        self.lock_path = self.path.with_suffix('.lock')
        self.limit = limit
        self.margin = margin
        self.label = None
        self.stop = threading.Event()
        self.thread = None
        self.mutex = threading.RLock()
        self.reservation = None

    def _read(self):
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state['limit_seconds'] != self.limit:
                raise ValueError('Persistent GPU budget cannot be reset or changed')
            return state
        return {'limit_seconds':self.limit,'charged_seconds':0.,'by_arm':{},'sessions':[],'active':None}

    def _mutate(self, fn):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.mutex, self.lock_path.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state = self._read()
            result = fn(state)
            write(self.path,state)
            return result

    def remaining(self):
        def calculate(s):
            extra = max(0., time.time()-s['active']['last_wall']) if s['active'] else 0.
            return self.limit-s['charged_seconds']-extra
        return self._mutate(calculate)

    def guard(self, estimate, *, checkpoint=False):
        if self.remaining() < estimate + (0 if checkpoint else self.margin):
            raise BudgetStop('Not enough reserved GPU wall time for next unit and checkpoint margin')

    def begin(self, label):
        def start(s):
            if s['active']:
                active = s['active']
                try:
                    os.kill(active['pid'],0)
                    raise RuntimeError('Another accounted GPU session is still active')
                except ProcessLookupError:
                    # Conservative crash accounting: charge through the last reserved
                    # unit deadline or heartbeat, whichever is later (capped at now).
                    end = min(time.time(), max(active['last_wall']+2.,active.get('reserved_until',active['last_wall'])))
                    extra = max(0,end-active['last_wall'])
                    s['charged_seconds'] += extra
                    s['by_arm'][active['label']] += extra
                    s['sessions'].append({**active,'crashed':True,'recovered_wall':time.time(),'extra_charge':extra})
                    s['active'] = None
            if self.limit-s['charged_seconds'] < self.margin:
                raise BudgetStop('Shared GRPO GPU budget exhausted')
            now = time.time()
            s['active'] = {'pid':os.getpid(),'label':label,'start_wall':now,'last_wall':now,'reserved_until':now}
            s['by_arm'].setdefault(label,0.)
        self._mutate(start)
        self.label = label
        self.stop.clear()
        def heartbeat():
            while not self.stop.wait(1.):
                self.tick()
                if self.remaining() <= 2.:
                    # Last-resort hard budget enforcement. Normal guards stop and
                    # checkpoint well before this deadline.
                    os.kill(os.getpid(),signal.SIGTERM)
        self.thread = threading.Thread(target=heartbeat,daemon=True)
        self.thread.start()
        return self

    def reserve_unit(self, estimate):
        self.guard(estimate)
        def reserve(s):
            if s['active'] and s['active']['pid']==os.getpid():
                s['active']['reserved_until']=time.time()+estimate
        self._mutate(reserve)

    def tick(self):
        def charge(s):
            a=s['active']
            if a is None or a['pid']!=os.getpid():
                raise RuntimeError('GPU accounting lease lost')
            now=time.time();delta=max(0,now-a['last_wall'])
            s['charged_seconds']+=delta;s['by_arm'][a['label']]+=delta;a['last_wall']=now
        self._mutate(charge)

    def charged(self,label_prefix=None):
        def calculate(s):
            return sum(v for k,v in s['by_arm'].items() if label_prefix is None or k.startswith(label_prefix))
        return self._mutate(calculate)

    def end(self):
        if self.label is None:
            return
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.tick()
        def close(s):
            s['sessions'].append({**s['active'],'end_wall':time.time(),'crashed':False})
            s['active']=None
        self._mutate(close)
        self.label=None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.end()
