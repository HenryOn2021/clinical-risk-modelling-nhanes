# Apply the corrected repository to GitHub

Review the generated metrics and figures before publishing. The raw XPT files,
processed participant-level data and serialized model remain ignored.

From the corrected repository directory:

```bash
git status
git diff --stat
git add .
git commit -m "fix: validate and reproduce NHANES modelling pipeline"
git push origin main
```

Recommended checks before committing:

```bash
python -m compileall -q scripts src tests
ruff check .
ruff format --check .
pytest -q
```

The repository intentionally versions aggregate metrics, baseline outputs and
figures so viewers can inspect the published result without downloading
participant-level data.
