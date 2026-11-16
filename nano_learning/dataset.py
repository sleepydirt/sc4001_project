from __future__ import annotations
from collections import defaultdict
import json
import random
from .common import STUDY,rows,sha,digest

class Dataset:
    def __init__(self,split,*,allow_test=False):
        if split not in ('train','validation','test'):raise ValueError('Invalid split')
        if split=='test':
            if not allow_test or not (STUDY/'evaluation_lock.json').exists():
                raise PermissionError('Final-test labels/predictions are guarded until protocol and selections are locked')
        self.split=split
        from .evidence import split_path
        self.path=split_path(STUDY, split)
        self.records=rows(self.path)
        self.byid={r['id']:r for r in self.records}
        self.catalogs=json.loads((STUDY/'catalogs.json').read_text())
        self.pool=defaultdict(lambda:defaultdict(list))
        for r in self.records:self.pool[r['component']][r['request_group']].append(r)
        self.sha256=sha(self.path)
        self._orders={}

    def example(self,row):
        if isinstance(row,str):row=self.byid[row]
        paths=row.get('adapted_audio_paths',row['audio_paths'])
        prompt=self.catalogs[row['domain']]['text']+row.get('audio_role_prompt','')
        return {**row,'audio_path':paths[0],'audio_paths':paths,'_prompt':prompt,
            'completion_cap':512,'target_text':row['target_text']}

    def batch(self,step,components,seed=17):
        if self.split!='train':raise PermissionError('Optimization sampler may read only training data')
        if not components or any(c not in self.pool for c in components):raise ValueError('Invalid train subset')
        result=[]
        for slot in range(4):
            draw=step*4+slot;cycle,position=divmod(draw,len(components))
            cache=(tuple(components),cycle,seed)
            if cache not in self._orders:
                order=list(components);random.Random(f'a2t-components:{seed}:{cycle}').shuffle(order)
                self._orders={cache:order}
            component=self._orders[cache][position]
            rng=random.Random(f'a2t-variant:{seed}:{draw}:{component}')
            request=rng.choice(sorted(self.pool[component]))
            recording=rng.choice(sorted(self.pool[component][request],key=lambda r:r['id']))
            result.append(self.example(recording))
        return result
