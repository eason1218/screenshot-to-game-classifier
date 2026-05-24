# Contributing

This project began as a Machine Learning 2 course project by a four-person team, but issues, suggestions, and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/eason1218/screenshot-to-game-classifier.git
cd screenshot-to-game-classifier
pip install -r requirements.txt
pip install git+https://github.com/ChaoningZhang/MobileSAM.git   # for photo-of-screen mode
```

The repo ships `best_model.pth`, so you can run the demo right away:

```bash
python demo/app.py
```

Datasets and other weights are not in the repo (see `.gitignore`); rebuild them with the `data/` scripts — see [DATA_COLLECTION.md](DATA_COLLECTION.md).

## Project conventions

- **Run from the project root.** All scripts add the root to `sys.path` and use root-relative paths.
- **Keep the letterbox geometry in sync.** The same `LetterboxResize` must be used in `train_tf`, `eval_tf`, and the demo's `_transform` — changing one without the others silently hurts accuracy.
- **Changing the class set** means updating `config.CLASS_NAMES` + `config.NUM_CLASSES` (alphabetical, matching the `ImageFolder` order) **and retraining** (the FC head dimension changes).
- **Set `DATA_SOURCE=local`** when training/evaluating on the merged 17-class dataset.

## Repository layout

| Area | Folder | Owner (course) |
|------|--------|----------------|
| Data pipeline | `data/` | Data team |
| Model & training | `model/` | Model team |
| Demo & inference | `demo/` | Demo team |
| Shared config | `config.py` | All (coordinate changes) |

## Pull requests

1. Create a feature branch from `main`.
2. Make focused commits with clear messages.
3. Keep docs in sync (README / folder READMEs / `DATA_COLLECTION.md`) when behavior changes.
4. Open a pull request describing the change and its motivation.

## Code style

- Follow the style of the surrounding code — type hints, docstrings, descriptive names.
- Keep functions small and single-purpose; mirror the existing module structure.
