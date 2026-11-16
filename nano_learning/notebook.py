"""Build and execute the paper preparation notebook using saved measurements."""
import argparse
import os
import sys
from pathlib import Path
from .common import ROOT, write


def build():
    import nbformat as n
    cells=[]
    def md(s):cells.append(n.v4.new_markdown_cell(s))
    def code(s):cells.append(n.v4.new_code_cell(s))
    md('# Audio2Tool paper preparation\n\nRegenerate figures and tables from saved measurements. This notebook performs no training, downloads or model inference. `PAPER.tex` is kept unchanged; consult the claim register and discrepancies before incorporating these assets. All experiments use one training seed.')
    code("""from pathlib import Path
import os, sys, csv, html
ROOT = Path.cwd() if (Path.cwd() / 'PAPER.tex').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from nano_learning.reporting import build, DEST
from IPython.display import display, Markdown, Image, HTML
result = build()
def show_table(name, columns):
    records = list(csv.DictReader((DEST / 'tables' / (name + '.csv')).open()))
    display(HTML('<table><tr>' + ''.join('<th>'+html.escape(k)+'</th>' for k in columns) + '</tr>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in columns)+'</tr>' for r in records) + '</table>'))
def figure(name):
    display(Image(filename=str(DEST / 'figures' / (name + '.png'))))
print('Exports:', DEST)
""")
    md('## Dataset\n\nCounts distinguish recording entries, requests, source components and distinct audio files. Duration sums recording references (matching historical reporting); unique-file duration is exported separately. The filtered benchmark is a restricted cohort and is not speaker-disjoint.')
    code("show_table('dataset', ['tier','scope','entries','requests','audio_files','hours'])\nshow_table('splits', ['split','components','entries','requests','audio_files','hours'])\nprint(result['panel'])")
    md('## Warmups\n\nCompare only the 640-update full-validation endpoints, with ties toward the smallest pool. Both models select 10%/640. Full validation occurs at 256, 512 and 640; mini-validation occurs at intervening 64-update boundaries. Rows show Samsone then Mellow; columns show response NLL (trailing 32-update mean), full-validation macro exact, and schema validity. Colors and line styles identify 10%, 30%, and 100% pools. Mellow ties at zero exact accuracy; 10% is selected by the tie-break. Metric definitions: `docs/metrics.md`.')
    code("show_table('warmups', ['model','pool','step','macro_exact','schema_validity','selected','historical_parent_step'])\nfigure('warmups')")
    md('## +64 pilot\n\nThese are historical pilot **test** measurements. Samsone starts from 10%/640 (endpoint lineage 704). Mellow starts from **100%/256 (endpoint lineage 320)**. The corrected Mellow +64 configuration is disabled and unrun; its values cannot be inferred from these zeros or from the +1024 result.')
    code("show_table('pilot_test', ['model','arm','parent_pool','parent_updates','lineage_updates','macro_exact','binary','schema_validity'])\nfigure('pilot64_test')")
    md('## +1024 comparison\n\nHeadline comparisons use full validation at the endpoint. Dashed trajectories are **mini-validation: training diagnostic**—40 recording entries, 40 distinct requests, 36 source components. They are not reliable estimates of full-split or deployment performance. Branches start independently from 10%/640; lineage is 1664. Mellow has SFT only. Training minutes sum optimizer-step timings and exclude loading, evaluation and checkpoint I/O.')
    code("show_table('endpoints', ['model','arm','additional_updates','macro_exact','binary','argument_F1','schema_validity','training_minutes'])\nfigure('continuation1024')\nfigure('continuation1024_schema')\nfigure('continuation_endpoints')")
    md('## SFT-4096\n\nThe Samsone extension adds 3072 updates to +1024 SFT. It restores optimizer moments and scaler and resets the scheduler over the new segment. Full validation occurs every 512 additional updates; mini-validation every 64. Endpoint lineage is 4736. The mini panel remains a training diagnostic.')
    code("figure('sft4096')")
    md('## Appendix: +1024 training diagnostics\n\nLeft: response NLL for Samsone and Mellow SFT. Right: sampled KL estimator to the frozen reference for Samsone binary and typed GRPO. Curves use trailing 32-update means. Mellow has no +1024 GRPO run. KL is the recorded token-level estimator, without multiplication by the KL penalty coefficient; it is distinct from the GRPO objective.')
    code("figure('continuation1024_training')")
    md('## Qualitative appendix\n\nPost-hoc examples verified against saved prediction rows. IDs, gold references, raw outputs, source hashes and checkpoint provenance accompany the exports. These examples are not a representative sample.')
    code("display(Markdown((DEST / 'qualitative.md').read_text()))\nshow_table('provenance', ['run','parent_updates','lineage_updates','sampler_offset','precision','sha256'])")
    md('## Caption drafts and audit\n\nVector PDF, 320-DPI PNG, CSV and LaTeX fragments are under `paper/generated`. Numerical tolerances and evidence paths are recorded in the audit. Regenerating figures does not reproduce stochastic training.')
    code("display(Markdown((DEST / 'captions.md').read_text()))\nfrom nano_learning.audit import build_register\nbuild_register()\ndisplay(Markdown((ROOT / 'docs/paper_discrepancies.md').read_text()))")
    nb=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Audio2Tool reporting','language':'python','name':'audio2tool-report'}})
    path=ROOT/'nano_learning/overview.ipynb';n.write(nb,path);return path


def execute(path):
    import nbformat
    from nbclient import NotebookClient
    directory=ROOT/'paper/generated/jupyter'
    write(directory/'kernels/audio2tool-report/kernel.json',{'argv':[sys.executable,'-m','nano_learning.kernel','-f','{connection_file}'],
        'display_name':'Audio2Tool reporting','language':'python','env':{'PYTHONPATH':str(ROOT)}})
    os.environ['JUPYTER_PATH']=str(directory)
    os.environ['JUPYTER_RUNTIME_DIR']='/tmp/audio2tool-jupyter-runtime'
    os.environ['IPYTHONDIR']='/tmp/audio2tool-ipython'
    nb=nbformat.read(path,as_version=4)
    NotebookClient(nb,timeout=300,kernel_name='audio2tool-report',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,path)
    write(ROOT/'paper/generated/evidence/notebook_execution.json',{'executed':True,'training':False,'inference':False,'downloads':False,
        'code_cells':sum(c.cell_type=='code' for c in nb.cells)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build-only',action='store_true');args=parser.parse_args()
    path=build()
    if not args.build_only:execute(path)

if __name__=='__main__':main()
