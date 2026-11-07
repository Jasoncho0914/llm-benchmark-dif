"""Small LaTeX writer shared by the paper tables."""
import numpy as np
from .common import OUTPUT


def number(value,digits=1):
    return '--' if not np.isfinite(value) else f'{value:.{digits}f}'


def count(value):
    return f'{int(value):,}'.replace(',','{,}')


def pvalue(value):
    return '$<0.001$' if value<.001 else f'${value:.3f}$'


def escape(text):
    return str(text).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')


def write(filename,label,caption,headers,rows,alignment=None):
    alignment=alignment or 'l'+'r'*(len(headers)-1)
    lines=[r'\begin{table}[ht]',r'\centering',r'\caption{'+caption+'}',
           r'\label{'+label+'}',r'\small',r'\resizebox{\linewidth}{!}{%',
           r'\begin{tabular}{'+alignment+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    (OUTPUT/filename).write_text('\n'.join(lines)+'\n')
