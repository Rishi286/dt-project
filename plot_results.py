import csv, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_FILE = 'results/summary_adaptive.csv'

rows = []
with open(DATA_FILE) as f:
    for r in csv.DictReader(f):
        rows.append({
            'allocator': r['allocator'],
            'regime':    r['regime'],
            'sla_rate':  float(r['sla_rate']),
            'avg_rt':    float(r['avg_rt']),
            'comp_rate': float(r['comp_rate']),
        })

ALLOCATORS = ['DT', 'AdaptiveDT', 'Threshold', 'Greedy']
REGIMES    = ['steady', 'diurnal', 'bursty']
COLORS     = {'DT':'#2563eb', 'AdaptiveDT':'#16a34a', 'Threshold':'#d97706', 'Greedy':'#7c3aed'}
LABELS     = {'DT':'DT', 'AdaptiveDT':'AdaptiveDT', 'Threshold':'Threshold', 'Greedy':'Greedy BF'}

def mean(vals): return sum(vals)/len(vals) if vals else 0

def get_means(metric):
    out = {}
    for regime in REGIMES:
        out[regime] = {}
        for alloc in ALLOCATORS:
            vals = [r[metric] for r in rows if r['regime']==regime and r['allocator']==alloc]
            out[regime][alloc] = mean(vals)
    return out

sla  = get_means('sla_rate')
rt   = get_means('avg_rt')
comp = get_means('comp_rate')

fig, axes = plt.subplots(3, 3, figsize=(11, 8))
fig.patch.set_facecolor('white')

METRICS = [
    (sla,  'SLA Violation Rate', '(lower is better)', '{:.4f}', False),
    (rt,   'Mean Response Time (s)', '(lower is better)', '{:.1f}', False),
    (comp, 'Task Completion Rate', '(higher is better)', '{:.2f}', False),
]

x = np.arange(len(ALLOCATORS))
w = 0.6

for row_i, (data, ylabel, note, fmt, _) in enumerate(METRICS):
    for col_i, regime in enumerate(REGIMES):
        ax = axes[row_i][col_i]
        vals = [data[regime][a] for a in ALLOCATORS]
        colors = [COLORS[a] for a in ALLOCATORS]
        bars = ax.bar(x, vals, width=w, color=colors, linewidth=0)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[a] for a in ALLOCATORS], fontsize=7.5, rotation=15, ha='right')
        ax.tick_params(axis='y', labelsize=7.5)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#cccccc')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: fmt.format(v)))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#eeeeee', linewidth=0.7)

        if row_i == 0:
            ax.set_title(regime.capitalize(), fontsize=9, fontweight='600', pad=6)
        if col_i == 0:
            ax.set_ylabel(ylabel, fontsize=8, labelpad=6)

        # value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        fmt.format(v), ha='center', va='bottom', fontsize=6.5, color='#333333')

# legend
handles = [plt.Rectangle((0,0),1,1,color=COLORS[a],label=LABELS[a]) for a in ALLOCATORS]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
           frameon=True, edgecolor='#dddddd', bbox_to_anchor=(0.5, 0.01))

fig.suptitle('Resource Allocator Comparison — SimPy Simulation Results\n'
             '(means over 5 random seeds, 5,000 tasks per trial)',
             fontsize=10, fontweight='600', y=0.99)

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
out = 'results/results_chart.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
