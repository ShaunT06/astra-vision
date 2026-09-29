# Hugging Face Spaces (Docker SDK, free CPU tier) or any Docker host.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    ASTRA_STORAGE_DIR=/home/user/storage

# Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=user app ./app
COPY --chown=user models ./models
COPY --chown=user reports ./reports
USER user

# Bake model weights into the image so the first request isn't a 1.5 GB download.
RUN python -c "from app.models import get_classifier, get_ood_head; [get_classifier(k) for k in ('siglip-probe','effnet-probe')]; get_ood_head()" \
 && python -c "from app.detector import _owl; _owl()"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
