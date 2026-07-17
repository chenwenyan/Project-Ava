# Repository Guidelines

## Project Structure & Module Organization

Core agentic video-analysis logic lives in `AVA/`, including graph storage, event/entity models, retrieval, and tree search. Model adapters are under `llms/`; embedding implementations are under `embeddings/`; dataset adapters are under `dataset/`. Dataset metadata and download helpers live in `datas/<Dataset>/`, while generated frames, graphs, and other intermediates belong in the untracked `AVA_cache/` tree. Top-level entry points are `preprocess_videos.py`, `graph_construction.py`, `query_SA.py`, and `query_CA.py`. Operational and smoke-test utilities are in `scripts/`; documentation images are in `arts/`.

## Build, Test, and Development Commands

Use Python 3.9 in an isolated environment:

```bash
conda create -n ava python=3.9
conda activate ava
pip install -r requirements.txt
python preprocess_videos.py --dataset lvbench --num_threads 10
python graph_construction.py --model qwenvl --dataset lvbench --video_id 1 --gpus 1
python query_SA.py --model qwenlm --dataset lvbench --video_id 1 --question_id 0 --gpus 1
python query_CA.py --model qwenvl --dataset lvbench --video_id 1 --question_id 0 --gpus 1
```

SA output must exist before running CA. Dataset download scripts, such as `datas/LVBench/download.sh`, fetch raw videos. Treat the longer pipeline scripts as machine-specific: inspect their absolute paths before use.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and lowercase module names where practical. Keep model and dataset selection centralized in `llms/init_model.py` and `dataset/init_dataset.py`. Prefer explicit CLI arguments and repository-relative paths. No formatter or linter is configured, so keep imports grouped, avoid unrelated reformatting, and follow nearby code.

## Testing Guidelines

There is no formal unit-test suite or coverage threshold. Validate focused changes with the smallest relevant dataset/video invocation. Model-loading smoke checks live in `scripts/test_qwenlm_load.py`, `scripts/test_qwenvl_load.py`, and `scripts/test_qwenvl_image.py`; these require model files and usually GPUs. Do not commit generated `outputs/`, `results/`, cached frames, graphs, or downloaded videos.

## Commit & Pull Request Guidelines

History uses short imperative subjects such as `Update README.md`; use a more specific equivalent, for example `Fix VideoMME question indexing`. Keep each commit scoped to one concern. Pull requests should explain the motivation, list commands run, identify required models/datasets and hardware, link relevant issues, and include sample output or screenshots when results or documentation visuals change.
