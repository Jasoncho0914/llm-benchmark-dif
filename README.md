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

If complete inputs already exist in `rawdata/`, the notebook uses them without downloading or changing them. Otherwise, it downloads the evaluation logs and metadata from Hugging Face, regrades MATH answers with Math-Verify, and builds the inputs there. Downloads can resume after interruption; temporary download records are removed after a successful build.

Downloading requires a Hugging Face account with access to the restricted evaluation-log datasets and a read token. The notebook uses a saved local Hugging Face login or asks for the token through a hidden prompt. **Do not paste tokens into notebook cells, source files, or this repository.** AWS credentials are not needed.

`codes/source_manifest.json` records the model names, evaluation dates, and question-version identifiers used to retrieve the analysis inputs. It does not contain question responses or correctness matrices.

## Verification status

The notebook analyses have been run and checked using the supplied curated inputs. Small fresh-download checks matched those inputs, including regraded MATH answers. A complete fresh download and rebuild has not yet been verified end to end. The notebook's final cell checks the main numerical results and the expected paper outputs.

## Measured analysis resources

The complete notebook analysis, starting from the curated inputs, ran on an 8-core Apple M1 machine with 16 GiB of RAM, with numerical libraries limited to one thread. The latest run took 2.32 minutes and reached 1.28 GiB peak process memory. Inputs, code, and generated results occupy approximately 0.7 GB of working storage. These measurements exclude downloading and rebuilding the source logs and installing the Python environment.
