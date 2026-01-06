# REM: Reasoning over Embodied Multi-Frame Trajectories

[![Paper](https://img.shields.io/badge/arXiv-2512.00736-b31b1b.svg)](https://arxiv.org/abs/2512.00736)
[![Conference](https://img.shields.io/badge/COLM-2025-blue.svg)](https://colmweb.org/)

Official implementation of **REM**, a benchmark for evaluating multimodal large language models on embodied spatial reasoning tasks.

> **REM: Evaluating LLM Embodied Spatial Reasoning through Multi-Frame Trajectories**  
> Jacob Thompson, Emiliano Garcia-Lopez, Yonatan Bisk  
> Carnegie Mellon University  
> *Published at COLM 2025*

## Abstract

Humans build viewpoint-independent cognitive maps through navigation, enabling intuitive reasoning about object permanence and spatial relations. We argue that multimodal large language models (MLLMs), despite extensive video training, lack this fundamental spatial reasoning capability. To demonstrate these limitations, we introduce **REM** (Reasoning over Embodied Multi-Frame Trajectories), a benchmark using controllable 3D environments for long-horizon embodied spatial reasoning.

REM systematically evaluates:
- **Object Counting** — "How many blue objects are there?"
- **Comparison** — "Are there more red cones or blue spheres?"
- **Relative Positioning** — "Is the green sphere to the left or right of the blue cuboid?"
- **Temporal Ordering** — "Did you see the red cone before or after the blue sphere?"

## Repository Structure

```
src/
├── data_generation/
│   ├── create_scene_config.py       # Generate single scene configuration
│   ├── create_scene_configs_batch.py # Batch generate (edit constants in file)
│   ├── render_scene.py              # Blender script to render trajectory
│   ├── render_scenes_batch.py       # Batch render multiple trajectories
│   └── question_generation/
│       ├── common_utils.py
│       ├── generate_number.py
│       ├── generate_comparison_in_frame.py
│       ├── generate_comparison_out_of_frame.py
│       ├── generate_directional.py
│       └── generate_order_preserving.py
├── inference/
│   ├── gemini.py                    # Google Gemini (parallel)
│   ├── llama.py                     # LLaMA via OpenRouter (parallel)
│   ├── openai_batch.py              # OpenAI Batch API
│   └── anthropic_batch.py           # Anthropic Batch API
└── analysis/
    ├── aggregate_results.py         # Aggregate Q&A results → master CSV
    └── compute_success_rates.py     # Compute accuracy (requires aggregated CSV)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  1. create_scene_config.py        → scene_config.json, trajectory_moves.txt     │
│                                            ↓                                    │
│  2. render_scene.py (Blender)     → images/*.png, annotations.csv               │
│                                            ↓                                    │
│  3. generate_*.py (question gen)  → {question_type}_questions.csv (with answer) │
│                                            ↓                                    │
│  4. inference/*.py                → adds model column to each question CSV      │
│                                            ↓                                    │
│  5. aggregate_results.py          → all_questions_aggregated.csv                │
│                                            ↓                                    │
│  6. compute_success_rates.py      → prints accuracy tables                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**CSV evolution:**

| Stage | CSV Contents |
|-------|--------------|
| After question gen | `question`, `answer` |
| After inference | `question`, `answer`, `gemini-2.0-flash`, `gpt-4o`, ... |
| After aggregation | All trajectories merged with metadata columns |

## Installation

**Prerequisites:**
- Python 3.9+
- [Blender 4.2+](https://www.blender.org/download/) (must be in PATH or set `BLENDER_EXE` in `render_scenes_batch.py`)

```bash
git clone https://github.com/jbt-cmu/REM.git
cd REM
pip install -r requirements.txt
```

**API Keys** (set as environment variables):
```bash
export GOOGLE_API_KEY="..."      # For Gemini
export OPENAI_API_KEY="..."      # For GPT / OpenAI Batch
export ANTHROPIC_API_KEY="..."   # For Claude Batch
export OPENROUTER_API_KEY="..."  # For LLaMA
```

## Usage

> **Important:** All scripts assume they are run from the repository root and use hardcoded paths to `src/trajectories/`. Generated data goes into `src/trajectories/{id}-run/`.

### 1. Generate Scene Configurations

Create a single scene with trajectory:

```bash
python src/data_generation/create_scene_config.py \
    --num-shapes 24 \
    --trajectory-size 8 \
    --seed 42 \
    --same-size \
    --homogenous-texture
```

**Key arguments:**
| Argument | Description |
|----------|-------------|
| `--num-shapes` | Total objects in scene (default: 24) |
| `--trajectory-size` | Number of image frames in the trajectory (2, 4, 8, 16, 32, or 64) |
| `--times` | Duplication pattern, e.g., `0 1` means one unique + one appearing twice |
| `--seed` | Random seed for reproducibility |
| `--same-size` | All objects are 1m scale (large only) |
| `--homogenous-texture` | All objects have rubber texture |
| `--bbox` | Half-size of trajectory bounding box (default: 6) |

**Output:** Creates `src/trajectories/{id}-run/` containing:
- `scene_config.json` — Scene and object definitions
- `trajectory_moves.txt` — Agent movement sequence (straight + turn moves)
- `scene_plot.png` — Top-down visualization

**Batch generation:** Edit constants in `create_scene_configs_batch.py` (`TRAJECTORY_SIZES`, `NUM_SHAPES_LIST`, `NUM_EXAMPLES`, `ALLOWED_TIMES_SETS`) then run:

```bash
cd src/data_generation
python create_scene_configs_batch.py
```

### 2. Render Trajectories in Blender

Render a single trajectory:

```bash
blender --background --python src/data_generation/render_scene.py -- \
    --config-file src/trajectories/0001-run/
```

**Output:** Creates `src/trajectories/{id}-run/images/` containing:
- `image_0000_init.png`, `image_0001_move_forward_1m.png`, ...
- `annotations.csv` — Per-frame visible objects with pixel coverage percentages

Batch render all trajectories:

```bash
python src/data_generation/render_scenes_batch.py
```

Optional: `--resume 0050` to skip trajectories before index 50.

### 3. Generate Questions

**Must run BEFORE inference.** Each script scans all `src/trajectories/*-run/images/annotations.csv` files and creates question CSVs in the same directories.

```bash
# From repository root:
python src/data_generation/question_generation/generate_number.py
python src/data_generation/question_generation/generate_comparison_in_frame.py
python src/data_generation/question_generation/generate_comparison_out_of_frame.py
python src/data_generation/question_generation/generate_directional.py
python src/data_generation/question_generation/generate_order_preserving.py
```

**Output per trajectory:**
- `number_questions.csv`
- `comparison_questions_in_frame.csv`
- `comparison_questions_out_of_frame.csv`
- `left_right_questions.csv`
- `order_preserving_questions.csv`

### 4. Run Inference

Inference scripts scan `src/trajectories/` for question CSVs, query the model with each question + trajectory images, and **append model responses as new columns** to the same CSVs.

```bash
# Gemini (parallel, immediate)
python src/inference/gemini.py

# LLaMA via OpenRouter (parallel, immediate)
python src/inference/llama.py

# OpenAI Batch API (async, cheaper, check back later)
python src/inference/openai_batch.py

# Anthropic Batch API (async, cheaper, check back later)
python src/inference/anthropic_batch.py
```

**Note:** Batch APIs (OpenAI, Anthropic) submit jobs asynchronously. You'll need to poll or retrieve results separately.

### 5. Analyze Results

**Step 1:** Aggregate all question CSVs into a single master file:

```bash
python src/analysis/aggregate_results.py
```

This produces `src/trajectories/all_questions_aggregated.csv` (or similar) with all Q&A pairs merged.

**Step 2:** Compute success rates. Edit `compute_success_rates.py` to set:
- `CSV_FILE` — path to aggregated CSV
- `MODEL_LIST` — column names for models you ran

```bash
python src/analysis/compute_success_rates.py
```

## Datasets

REM consists of three datasets:

| Dataset | Purpose | Trajectories | Q&A Pairs |
|---------|---------|--------------|-----------|
| **Baseline** | Comprehensive evaluation across varied complexity | 3,000+ | ~50,000 |
| **Single Frame** | Control for single-image counting | — | — |
| **Full Rotation** | Stress test for object permanence under 360° rotation | — | — |

The Baseline dataset varies:
- **Trajectory length:** 2, 4, 8, 16, 32, or 64 frames
- **Scene congestion:** 8, 16, 24, 36, or 48 objects
- **Object duplication:** All-unique to mostly-duplicated

## Citation

```bibtex
@inproceedings{thompson2025rem,
  title={REM: Evaluating LLM Embodied Spatial Reasoning through Multi-Frame Trajectories},
  author={Thompson, Jacob and Garcia-Lopez, Emiliano and Bisk, Yonatan},
  booktitle={Conference on Language Modeling (COLM)},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
