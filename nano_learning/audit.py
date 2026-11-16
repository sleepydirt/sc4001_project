"""Claim-to-evidence register pinned to the author-updated manuscript."""
import csv
import json
import re
from pathlib import Path
from .common import ROOT, STUDY, write, sha
from .evidence import roots

COMMAND='.venv/bin/python -m nano_learning.notebook'


def read_table(name):
    return list(csv.DictReader((ROOT/'paper/generated/tables'/f'{name}.csv').open()))


def numeric_rows(text,label):
    position=text.index('\\label{'+label+'}')
    block=text[text.rfind(r'\begin{table',0,position):position]
    block=re.search(r'\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}',block,re.S)[1]
    block=re.sub(r'\\(?:toprule|midrule|bottomrule)', '', block)
    result=[]
    for line in re.split(r'\\\\',block):
        if '&' not in line:continue
        line=re.sub(r'\\textbf\{([^{}]*)\}',r'\1',line).replace(',','')
        values=[]
        for field in line.split('&'):
            field=field.strip()
            match=re.match(r'^(-?\d+(?:\.\d+)?)',field)
            values.append(float(match[1]) if match else None)
        if any(v is not None for v in values):result.append(values)
    return result


def numeric_audit(text):
    checks=[]
    def compare(label, actual, expected, tolerances):
        if len(actual)!=len(expected):raise ValueError(f'{label}: unexpected manuscript row count')
        for i,(a,b) in enumerate(zip(actual,expected),1):
            for j,(x,y) in enumerate(zip(a,b)):
                if y is None:continue
                tolerance=tolerances[j] if isinstance(tolerances,list) else tolerances
                checks.append(dict(table=label,row=i,column=j+1,paper=x,evidence=y,tolerance=tolerance,
                    matches=x is not None and abs(x-y)<=tolerance+1e-12))
    warm=read_table('warmups')
    expected=[[None,float(r['pool'])*100,640,float(r['macro_exact'])*100,float(r['schema_validity'])*100] for r in warm]
    label='tab:a2t-warmup-selection';compare(label,numeric_rows(text,label),expected,[0,0,0,.005,.005])
    pilot=read_table('pilot_test');expected=[[None,None,int(r['lineage_updates']),*[float(r[k])*100 for k in ('macro_exact','binary','schema_validity')]] for r in pilot]
    label='tab:a2t-pilot-test';compare(label,numeric_rows(text,label),expected,[0,0,0,.005,.005,.005])
    endpoints=read_table('endpoints'); parent=warm[0]
    expected=[[None,None,640,*[float(parent[k])*100 for k in ('macro_exact','binary','argument_F1','schema_validity')],None]]
    expected.extend([[None,None,640+int(r['additional_updates']),*[float(r[k])*100 for k in ('macro_exact','binary','argument_F1','schema_validity')],float(r['training_minutes'])] for r in endpoints])
    label='tab:a2t-long-dev';compare(label,numeric_rows(text,label),expected,[0,0,0,.005,.005,.005,.005,.05])
    expected=[[float(r['pool'])*100,int(r['components']),int(r['request_groups']),int(r['recordings']),float(r['hours'])] for r in warm[:3]]
    label='tab:a2t-warmup-design';compare(label,numeric_rows(text,label),expected,[0,0,0,0,.00005])
    splits=read_table('splits'); total=sum(int(r['components']) for r in splits)
    expected=[[None,int(r['components']),100*int(r['components'])/total,int(r['entries']),int(r['requests']),int(r['audio_files']),float(r['hours'])] for r in splits]
    expected.append([None,total,100,sum(int(r['entries']) for r in splits),sum(int(r['requests']) for r in splits),sum(int(r['audio_files']) for r in splits),sum(float(r['hours']) for r in splits)])
    label='tab:audio2tool-grouped-split';compare(label,numeric_rows(text,label),expected,[0,0,.05,0,0,0,.005])
    dataset=read_table('dataset')
    for scope,label in [('source','tab:audio2tool-hours'),('filtered','tab:audio2tool-filtered')]:
        rr=[r for r in dataset if r['scope']==scope]
        expected=[[int(r['tier']),*([None] if scope=='source' else []),int(r['requests']),int(r['audio_files']),float(r['hours'])] for r in rr]
        expected.append([None,*([None] if scope=='source' else []),sum(int(r['requests']) for r in rr),sum(int(r['audio_files']) for r in rr),sum(float(r['hours']) for r in rr)])
        tolerances=([0,0,0,0,.05] if scope=='source' else [0,0,0,.005])
        compare(label,numeric_rows(text,label),expected,tolerances)
    parameters=read_table('parameters');keys=['decoder_trainable','projector_trainable','extra_separator_trainable','trainable_total','frozen_encoder','model_total']
    expected=[[None,*[int(r[k]) for r in parameters]] for k in keys]
    label='tab:parameter-counts';compare(label,numeric_rows(text,label),expected,0)
    return checks


def document_checks(text):
    """Static checks only; a clean result is not a successful TeX compilation."""
    clean=re.sub(r'(?<!\\)%[^\n]*','',text)
    labels=set(re.findall(r'\\label\{([^}]+)\}',clean))
    refs=set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}',clean))
    cites={key.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',clean) for key in group.split(',')}
    bibliography=[name.strip()+'.bib' for group in re.findall(r'\\bibliography\{([^}]+)\}',clean) for name in group.split(',')]
    existing=[ROOT/name for name in bibliography if (ROOT/name).is_file()]
    supplied=ROOT/'paper/references.bib'
    if supplied.exists() and supplied not in existing:existing.append(supplied)
    keys={key for p in existing for key in re.findall(r'@\w+\s*\{\s*([^,\s]+)',p.read_text())}
    graphics=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',clean)
    import shutil
    return dict(undefined_references=sorted(refs-labels),cited_keys=len(cites),
        missing_citation_keys=sorted(cites-keys),bibliography_paths=bibliography,
        missing_bibliography_paths=[p for p in bibliography if not (ROOT/p).is_file()],
        supplied_bibliography='paper/references.bib',
        missing_graphics=[p for p in graphics if not (ROOT/p).is_file()],
        local_neurips_style_present=(ROOT/'neurips_2026.sty').exists(),
        xelatex_available=bool(shutil.which('xelatex')),lualatex_available=bool(shutil.which('lualatex')),
        compiled=False)


def build_register():
    review=json.loads((ROOT/'nano_learning/resources/paper_review.json').read_text())
    paper=ROOT/'PAPER.tex';expected=review['paper_sha256']
    if sha(paper)!=expected:
        raise ValueError('PAPER.tex differs from the reviewed draft; re-investigate claims before updating the review baseline')
    text=paper.read_text();records=review['claims']
    for r in records:
        if r['paper_anchor'] not in text:raise ValueError('Claim anchor absent: '+r['id'])
        r['line']=text[:text.index(r['paper_anchor'])].count('\n')+1
        r['reproduce']=COMMAND
        for path in r['evidence']:
            if not (ROOT/path).is_file():raise FileNotFoundError(path)
    checks=numeric_audit(text);mismatches=[x for x in checks if not x['matches']]
    structure=document_checks(text)
    doc=ROOT/'docs';doc.mkdir(exist_ok=True)
    payload=dict(paper_sha256=expected,previous_paper_sha256=review['previous_paper_sha256'],
        reviewed_on=review['reviewed_on'],claims=records,numeric_checks=checks,document_checks=structure)
    write(doc/'claim_evidence.json',payload)
    fields=['id','line','status','previous_status','change','priority','paper_anchor','assessment','remedy','evidence','external_sources','reproduce']
    with (doc/'claim_evidence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()})
    lines=['# Important remaining paper discrepancies','', 'Review of the latest author-updated draft. Placeholder figures are excluded. The manuscript was not edited.','']
    for r in records:
        lines += [f"## {r['id']} (PAPER.tex:{r['line']})",'',r['assessment'],'',f"**Suggested correction:** {r['remedy']}",'']
    lines += ['These corrections require no new training. Evaluation hardware should be substantiated from its actual execution records.']
    (doc/'paper_discrepancies.md').write_text('\n'.join(lines)+'\n')
    (doc/'claim_evidence.md').write_text('# Current paper review\n\nSee [important remaining discrepancies](paper_discrepancies.md). Numerical checks and source references are in claim_evidence.json.\n')
    return dict(claims=len(records),numeric_cells=len(checks),discrepancies=mismatches,document_checks=structure)

if __name__=='__main__':print(json.dumps(build_register(),indent=2))
