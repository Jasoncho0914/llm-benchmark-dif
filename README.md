# Differential Item Functioning as an Item-Level Diagnostic for LLM Benchmarks

This repository contains the reproducibility code for **Differential Item Functioning as an Item-Level Diagnostic for LLM Benchmarks**. The analysis uses question-level Open LLM Leaderboard v2 evaluation logs for BBH, GPQA, IFEval, MATH Level 5, MMLU-Pro, and MuSR.

## Run the analysis

Use Python 3.12 and install the pinned dependencies:

```bash
python -m pip install -r codes/requirements.txt
```

Open `reproduce.ipynb`, select that Python environment as its kernel, and run all cells from top to bottom. No GPU or new model inference is required.

- `codes/` contains the download, data preparation, and analysis functions.
- `reproduce.ipynb` prepares the inputs and runs the paper analyses; its saved outputs show the completed results.
- `reproduce.html` is a rendered version with all paper tables and embedded figures. Open it in a browser to view the results without running Python.
- `rawdata/` holds local input data and is excluded from Git.
- `output/` contains the paper's final LaTeX tables and figure files. Running the analysis replaces these generated files.

To rerun the notebook and refresh both saved notebook outputs and the HTML copy:

```bash
python -m codes.reporting
```

## Data access

We obtain question-level evaluation logs and model metadata from the [Open LLM Leaderboard v2 on Hugging Face](https://huggingface.co/open-llm-leaderboard). Run `reproduce.ipynb` from top to bottom to download these files, prepare the analysis inputs in `rawdata/`, and generate the paper's tables and figures. Data preparation includes grading MATH responses with Math-Verify. If the inputs are already present, the notebook reuses them.

Downloading from Hugging Face requires an account with access to the leaderboard's restricted evaluation-log datasets and a read token. The notebook uses your saved Hugging Face login or asks for the token through a hidden prompt.

A copy of the prepared `rawdata/` folder is also available on [Google Drive](https://drive.google.com/file/d/1MglQxb9qink08iKhzKJG42MOeT_8DJ_2/view?usp=sharing). To use this copy, extract the archive into the repository root and run the notebook.

## Measured analysis resources

The complete notebook analysis, starting from the curated inputs, ran on an 8-core Apple M1 machine with 16 GiB of RAM, with numerical libraries limited to one thread. The latest run took 2.32 minutes and reached 1.28 GiB peak process memory. Inputs, code, and generated results occupy approximately 0.7 GB of working storage. These measurements exclude downloading and rebuilding the source logs and installing the Python environment.
