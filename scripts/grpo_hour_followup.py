import datetime,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/grpo_attempt_20261107';deadline=datetime.datetime(2026,11,7,9,22,35,tzinfo=datetime.timezone.utc).timestamp()
def run(script,log,*args):
    remaining=int(deadline-time.time()-10)
    if remaining<90:return False
    with (out/log).open('w') as f:
        return subprocess.call(['timeout','--signal=TERM','--kill-after=5s',str(remaining)+'s',str(root/'.venv/bin/python'),'-u',str(root/'scripts'/script),*args],stdout=f,stderr=subprocess.STDOUT,cwd=root)==0
while not (out/'result.json').exists() and time.time()<deadline-600:time.sleep(5)
if not (out/'result.json').exists():raise SystemExit('First run did not finish within the shared deadline')
run('grpo_compare.py','comparison_console.log')
run('grpo_weight_check.py','weight_console.log')
run('report_grpo_hour.py','report_console.log')
r=json.loads((out/'result.json').read_text())
flat=all(r.get('grpo_final',{}).get(k,{}).get('exact',0)<=r.get('common',{}).get(k,{}).get('exact',0) for k in ['train','dev_id','dev_ood'])
if flat and time.time()<deadline-1200:
    if run('grpo_hour_strong.py','strong_init_console.log') and (out/'strong_init/result.json').exists():
        run('grpo_compare.py','strong_comparison_console.log','strong_init')
        run('grpo_weight_check.py','strong_weight_console.log','strong_init')
        run('report_grpo_hour.py','strong_report_console.log','strong_init')
