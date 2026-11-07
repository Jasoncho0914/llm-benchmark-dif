"""Only the six figures included in the paper are written to output/."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .common import OUTPUT

def p_latex(p):
    return '$<0.001$' if p<.001 else f'${p:.3f}$'

def category_figure(benchmark, details, summaries, filename):
    group = details[(details.benchmark == benchmark) & details.included_in_tests].copy()
    group = group.sort_values(['C_pct','category'])
    row = summaries.set_index('benchmark').loc[benchmark]
    height = max(5, .34*len(group)+1.6)
    fig, axes = plt.subplots(1, 2, figsize=(13, height), sharey=True)
    positions = np.arange(len(group))
    axes[0].barh(positions, group.C_pct, color='#999999', edgecolor='black', linewidth=.5)
    axes[1].barh(positions, group.C_favor_Llama_pct, color='#d62728', edgecolor='black', linewidth=.5)
    axes[1].barh(positions, group.C_favor_Qwen_pct, left=group.C_favor_Llama_pct,
                 color='#2ca02c', edgecolor='black', linewidth=.5)
    axes[0].set_yticks(positions, labels=group.category.str.replace('_',' ',regex=False))
    axes[1].tick_params(axis='y', labelleft=False)
    axes[0].set_title(f'{benchmark} — DIF prevalence', fontsize=14)
    axes[1].set_title(f'{benchmark} — direction among Category C', fontsize=14)
    axes[0].set_xlabel('Category C (%)')
    axes[1].set_xlabel('Category-C items (%)')
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_xticks([0,20,40,60,80,100])
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
    axes[1].axvline(50,color='black',ls=':',lw=1)
    axes[0].text(.98,-.08,r'$\chi^2$ $p$ '+p_latex(row.class_p),transform=axes[0].transAxes,ha='right')
    axes[1].text(.98,-.08,r'$\chi^2$ $p$ '+p_latex(row.direction_p),transform=axes[1].transAxes,ha='right')
    fig.legend(handles=[Patch(facecolor='#d62728',label='Favors Llama'),
                        Patch(facecolor='#2ca02c',label='Favors Qwen')],
               loc='upper center',bbox_to_anchor=(.75,1),ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.035,1,.95))
    for ext in ['pdf']:
        fig.savefig(OUTPUT/f'{filename}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def plot_selection(table):
    with plt.rc_context({'font.size':12, 'pdf.fonttype':42}):
        fig, ax = plt.subplots(figsize=(8,4.8),layout='constrained')
        y = np.arange(len(table))
        values = table.adjusted_difference_pp.to_numpy()
        ax.errorbar(values,y,xerr=np.vstack([values-table.low_pp,table.high_pp-values]),
                    fmt='o',color='#287a52',capsize=3,markersize=6)
        ax.set_yticks(y,[f'{r.task.replace("_"," ")} ({r.answer})' for r in table.itertuples()])
        ax.invert_yaxis(); ax.axvline(0,color='.4',linestyle='--',linewidth=1)
        ax.set_xlabel('Qwen − Llama: adjusted answer-selection rate (pp)')
        ax.set_title('BBH'); ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        for ext in ['pdf']:
            fig.savefig(OUTPUT/f'fig_bbh_answer_selection.{ext}',dpi=220)
        plt.close(fig)


COLORS={'All questions':'#555555','Favor Qwen':'#2ca02c','Favor Llama':'#d62728'}


def overall_gaps(summary):
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True)
    order=['All questions','Favor Llama','Favor Qwen']
    for ax,b in zip(axes,['bbh','musr']):
        group=summary[summary.benchmark.eq(b)].set_index('pool').loc[order]
        for index,(pool,row) in enumerate(group.iterrows()):
            ax.vlines(index,row.ci_low,row.ci_high,color=COLORS[pool],lw=1.6)
            ax.plot(index,row.gap_pp,'o',color=COLORS[pool])
            ax.hlines([row.ci_low,row.ci_high],index-.04,index+.04,color=COLORS[pool])
        ax.axhline(0,color='black',lw=1)
        ax.axhline(group.loc['All questions','gap_pp'],color='#888888',ls=':')
        ax.set_xticks(range(3),labels=order,rotation=20,ha='right')
        ax.set_title('BBH' if b=='bbh' else 'MuSR');ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Qwen − Llama matched accuracy gap (pp)')
    fig.tight_layout();fig.savefig(OUTPUT/'overall_rq3.png',dpi=190);plt.close(fig)


def fold_gap_curves(curves):
    with plt.rc_context({'font.size':11,'axes.labelsize':11,'xtick.labelsize':10,
                         'ytick.labelsize':10,'axes.titlesize':11,'pdf.fonttype':42}):
        for b in ['bbh','musr']:
            fig,axes=plt.subplots(5,1,figsize=(4.2,7.7),sharex=True,sharey=True)
            for fold,ax in enumerate(axes,1):
                group=curves[(curves.benchmark==b)&(curves.fold==fold)]
                for pool in ['All questions','Favor Qwen','Favor Llama']:
                    z=group[group.pool.eq(pool)].sort_values('midpoint')
                    ax.plot(z.midpoint,z.gap_pp,'o-',ms=2,lw=1.15,color=COLORS[pool],label=pool)
                    ax.fill_between(z.midpoint,z.ci_low,z.ci_high,color=COLORS[pool],alpha=.13,lw=0)
                ax.axhline(0,color='black',ls='--',lw=.7);ax.set_ylim(-35,35)
                ax.set_yticks([-30,0,30]);ax.grid(alpha=.15)
                ax.spines[['top','right']].set_visible(False)
                ax.text(.025,.88,f'Fold {fold}',transform=ax.transAxes,fontsize=11,fontweight='bold',va='top')
                ax.tick_params(axis='both',pad=2)
            axes[-1].set_xlabel('Training matching score (%)',labelpad=4)
            fig.supylabel('Qwen − Llama accuracy gap (pp)',x=.018,fontsize=11)
            fig.suptitle('BBH' if b=='bbh' else 'MuSR',fontsize=15,y=.997,fontweight='bold')
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.966),ncol=2,
                       frameon=False,fontsize=10,columnspacing=1,handlelength=1.5,labelspacing=.3)
            fig.subplots_adjust(left=.17,right=.985,bottom=.085,top=.88,hspace=.18)
            fig.savefig(OUTPUT/f'{b}_gap_curves_large_text.pdf',bbox_inches='tight',pad_inches=.035)
            plt.close(fig)
