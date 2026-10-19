# RAG Assignment: Baseline vs. Re-Ranking

Retrieval-augmented question answering over the *Brass: Birmingham* board-game rulebook, comparing two retrieval pipelines that share one chunked corpus and one prompt:

- **Pipeline 1 (baseline):** bi-encoder `sentence-transformers/all-MiniLM-L6-v2` + FAISS → top-10 by cosine similarity → top-3 → LLM
- **Pipeline 2 (re-ranking):** the same top-10 → cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` → top-3 → LLM

Every model runs locally from Hugging Face. The generator is `Qwen/Qwen3-4B-Instruct-2507` with greedy decoding.

## Files

| Path | Contents |
|---|---|
| `RAG_Baseline_vs_ReRanking.ipynb` | The full notebook, saved with all outputs |
| `doc.pdf` | The rulebook (v2018.11, from roxley.com/brass-birmingham-rulebook) |
| `results/` | Tables (CSV/JSON) and charts written by the notebook |
| `requirements.txt` | Package versions the notebook was run with |
| `report/` | The report: `report.tex`, the compiled `report.pdf`, and vector versions of the charts in `figures/` |

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows; use `source .venv/bin/activate` elsewhere
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129   # skip for the CPU build
pip install -r requirements.txt
```

Open the notebook in VS Code or Jupyter, select this environment as the kernel, and run all cells.
The first run downloads about 8.5 GB of model weights.
A full run takes about 3 minutes on an RTX 5060 Ti. On CPU the retrieval steps are fine, but LLM generation is much slower.

To rebuild the report, run `tectonic report.tex` inside `report/`, or upload the folder to Overleaf.
