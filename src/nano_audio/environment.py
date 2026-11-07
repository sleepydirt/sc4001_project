import importlib,importlib.metadata,platform,shutil,subprocess,sys
from .utils import write_json

def inspect(output):
    import psutil
    result={'python':sys.version,'executable':sys.executable,'os':platform.platform(),'architecture':platform.machine(),'ram':psutil.virtual_memory()._asdict(),'disk':shutil.disk_usage('.')._asdict(),'packages':{}}
    for name in ['torch','torchaudio','torchvision','transformers','tokenizers','datasets','accelerate','vllm','torchcodec','kokoro']:
        try:
            module=importlib.import_module(name);result['packages'][name]={'version':importlib.metadata.version(name),'import':'ok'}
        except Exception as e:result['packages'][name]={'import':'failed','error':repr(e)}
    import torch
    result['cuda']={'available':torch.cuda.is_available(),'compiled_cuda':torch.version.cuda,'compiled_arches':torch.cuda.get_arch_list()}
    if torch.cuda.is_available():
        result['cuda'].update(name=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),bf16=torch.cuda.is_bf16_supported(),vram=torch.cuda.get_device_properties(0).total_memory)
        x=torch.randn(32,32,device='cuda',requires_grad=True);x.square().mean().backward();result['cuda']['backward_norm']=x.grad.norm().item()
        with torch.autocast('cuda',dtype=torch.float16):loss=(x@x).square().mean()
        scaler=torch.amp.GradScaler('cuda');scaler.scale(loss).backward();result['cuda']['fp16_autocast_backward']=bool(torch.isfinite(x.grad).all())
    result['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
    write_json(output,result);return result
