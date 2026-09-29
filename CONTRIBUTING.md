# Contributing

Thanks for your interest in ASTRA VISION.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd web && npm install
```

Windows users hitting a PyTorch DLL block (Smart App Control) should develop inside WSL; see the README.

## Before opening a pull request

```bash
pytest tests/test_validation.py    # fast, no model downloads (this is what CI runs)
pytest -q                          # full suite, downloads model weights on first run
cd web && npm run build            # front end must build
```

- Keep changes focused; one topic per pull request.
- Add or update tests for behaviour changes, especially input validation.
- If you retrain or change a model, regenerate `reports/` with `python -m scripts.evaluate` and update the README results table.
- Do not commit datasets, `.env` files or API keys (none are needed).

## Reporting issues

Use the issue templates. For misclassifications, include the image (if you may share it), the model used and the JSON response.
