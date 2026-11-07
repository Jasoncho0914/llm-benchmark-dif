"""Display paper tables/figures in the notebook and export an executed HTML copy."""
from html import escape
from io import BytesIO
from pathlib import Path
import re

_ENABLED = False


def enable():
    global _ENABLED
    _ENABLED = True


def _plain_latex(value):
    text = str(value).replace('{,}', ',')
    for old, new in [(r'\Delta', 'Δ'), (r'\beta', 'β'), (r'\alpha', 'α'), (r'\chi^2', 'χ²'), (r'\leq', '≤')]:
        text = text.replace(old, new)
    text = re.sub(r'\\(?:mathrm|textbf|textit)\{([^{}]*)\}', r'\1', text)
    for old, new in [(r'\%', '%'), (r'\_', '_'), (r'\&', '&'), (r'\#', '#')]:
        text = text.replace(old, new)
    return text.replace('$', '').replace('{', '').replace('}', '')


def show_table(label, caption, headers, rows):
    if not _ENABLED:
        return
    from IPython.display import HTML, display
    head = ''.join('<th>' + escape(_plain_latex(v)) + '</th>' for v in headers)
    body = ''.join('<tr>' + ''.join('<td>' + escape(_plain_latex(v)) + '</td>' for v in row) + '</tr>' for row in rows)
    display(HTML(
        '<div style="overflow-x:auto;margin:1em 0">'
        f'<table class="paper-table" id="{escape(label, quote=True)}" style="border-collapse:collapse;text-align:right">'
        f'<caption style="text-align:left;caption-side:top;padding:0.6em 0;font-weight:500">{escape(_plain_latex(caption))}</caption>'
        '<thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>'
    ))


def show_figure(figure):
    if not _ENABLED:
        return
    from IPython.display import Image, display
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=180, bbox_inches='tight')
    display(Image(data=buffer.getvalue(), width=min(1200, int(figure.get_figwidth() * 120))))


def show_examples(examples):
    if not _ENABLED:
        return
    from IPython.display import HTML, display
    cards = []
    for item in examples:
        cards.append(
            '<article class="paper-example" style="border:1px solid #ccc;padding:1em;border-radius:6px">'
            f'<h4>{escape(item["benchmark"])} · favors {escape(item["family"])}</h4>'
            f'<p>{escape(item["text"])}</p><p>Δ<sub>MH</sub> = {item["delta"]:+.2f}; {escape(item["detail"])}</p>'
            f'<p>Overall: Llama {item["overall_llama"]:.0f}%, Qwen {item["overall_qwen"]:.0f}%</p>'
            f'<p>Median stratum: Llama {item["median_llama"]:.0f}%, Qwen {item["median_qwen"]:.0f}% '
            f'({item["median_qwen"]-item["median_llama"]:+.0f} pp)</p></article>'
        )
    display(HTML('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1em">' + ''.join(cards) + '</div>'))


def execute_and_export():
    """Execute reproduce.ipynb, retain results, and embed all figures in reproduce.html."""
    import sys
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    root = Path(__file__).resolve().parents[1]
    path = root / 'reproduce.ipynb'
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=3600, resources={'metadata': {'path': str(root)}})
    client.execute(cleanup_kc=True)
    outputs = [output for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs]
    figures = sum('image/png' in output.get('data', {}) for output in outputs)
    tables = sum(output.get('data', {}).get('text/html', '').count('class="paper-table"') for output in outputs)
    examples = sum(output.get('data', {}).get('text/html', '').count('class="paper-example"') for output in outputs)
    if (tables, figures, examples) != (12, 6, 4):
        raise RuntimeError(f'Incomplete rendered results: {tables} tables, {figures} figures, {examples} examples.')
    nbformat.write(notebook, path)
    body, _ = HTMLExporter().from_notebook_node(notebook, resources={'metadata': {'name': 'reproduce'}})
    (root / 'reproduce.html').write_text(body)
    print(f'Saved executed notebook and HTML with {tables} paper tables, {figures} embedded figures and {examples} examples.')


if __name__ == '__main__':
    execute_and_export()
