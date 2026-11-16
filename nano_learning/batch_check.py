"""Verify variable-length catalog batching separately on each primary model."""
import gc
import json
import torch
from .common import STUDY,write,tracking
from .profile import training_examples
from .policies import load_policy
from .sampling import generate_batch
from .accounting import GPULedger

def main():
    torch.set_num_threads(4);examples=training_examples()
    for model in ('samsone','mellow'):
        run=tracking(model+'-batch-boundaries','engineering',{'training_only':True,'variable_catalog_lengths':True})
        ledger=GPULedger();ledger.begin('engineering-batch-'+model)
        try:
            policy=load_policy(model);chosen=[examples[0],examples[2],examples[4],examples[4]]
            config={'do_sample':False,'max_new_tokens':16}
            serial=[policy.generate(e,config) for e in chosen]
            batch=generate_batch(policy,chosen,config,force_batch=True)
            identical=all(a['ids']==b['ids'] for a,b in zip(serial,batch))
            path=STUDY/'interface_profile.json';profiles=json.loads(path.read_text())
            profiles[model]['variable_catalog_batch_checked']=True
            profiles[model]['variable_catalog_batch_matches_serial']=identical
            profiles[model]['batching_activated']=False
            profiles[model]['decoding']='native serial unconstrained, same for all arms and evaluation'
            profiles[model]['batch_boundary_examples']=[{'id':e['id'],'serial':a,'batched':b} for e,a,b in zip(chosen,serial,batch)]
            profiles[model]['batch_size']=1;write(path,profiles)
            run.log({'engineering/variable_catalog_batch_matches_serial':int(identical),'engineering/batching_activated':0})
            del policy;gc.collect();torch.cuda.empty_cache()
        finally:
            try:run.finish()
            finally:ledger.end()

if __name__=='__main__':main()
