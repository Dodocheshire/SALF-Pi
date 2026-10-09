"""Show published context separately from the released-model reproduction."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    data=json.loads((ROOT/'reports/released_baseline.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    names=['LF-CBM\npaper context','SALF-CBM\npaper context','SALF released\nlocal reproduction']
    bars=ax.bar(names,[71.95,75.32,100*data['accuracy']],color=['#64748B','#F59E0B','#0891B2'])
    ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.set_ylabel('ImageNet top-1 accuracy (%)');ax.set_ylim(0,90)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('SALF selection and baseline readiness\nNo trained noise adapter and no new gain claim',pad=15)
    fig.supxlabel('Paper values: CVPR 2025 Table 1, sparse models.\nLocal: released checkpoint, all 50,000 images / 1,000 classes.',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'reports/released_baseline.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
