# ASTRA VISION — Build Plan

**Challenge:** 02 — AI-Based Defence Object Recognition System (solo entry)
**Deadline:** 1 October 2026
**Constraint:** 100% free — open-weight models, no paid APIs, free hosting, free GPU (Kaggle/Colab) only if needed.
**Frontend:** to be decided later. The backend is built as a clean JSON API so any frontend can plug in.

---

## 0. Status — what changed during the build (29 Sep)

| Plan said | What was built | Why |
|---|---|---|
| Kaggle datasets + civilian class | **ASTRA starter set only** (5 classes), no civilian class | No time to download; civilian objects handled by the zero-shot OOD gate instead |
| Fine-tune EfficientNet-B0 | **Frozen backbone + logistic-regression head** for SigLIP 2 *and* EfficientNet-B0 | Only 117 clean images — full fine-tuning would overfit |
| 70/15/15 split | **Group-aware 5-fold CV × 3 repeats** | A 15% test split is ~18 images; groups stop photo series leaking |
| Main model EfficientNet-B0 | **SigLIP 2 + trained head** is the default | 99.2% vs 91.2% held-out accuracy |
| — | **Data audit**: 33/150 starter images excluded (`dataset/exclusions.csv`) | Maps, drone-shot landscapes, ceremonies, interiors mislabelled as objects |
| Plain Grad-CAM | Grad-CAM with **eigen-smoothing** for the ViT | Plain Grad-CAM highlighted background tokens on SigLIP |
| Run on Windows | Run in **WSL (Ubuntu)** | Windows Smart App Control blocks PyTorch DLLs |
| Weights in `models/` | Trained heads (20 KB each) + `thresholds.json` committed in `models/` | Small enough for git |

Done: MUST ✅ · SHOULD ✅ · BONUS ✅ (all 7). Remaining: frontend, deployment, video.

---

## 1. Model stack (all free, open licences)

### Core models (what we build with)

| Role | Model | Licence | Size | Why this one |
|---|---|---|---|---|
| **Main classifier** (trained on our data) | **EfficientNet-B0** via `timm` (ImageNet-pretrained) | Apache 2.0 | ~5.3M params | Small, generalises well from small datasets with transfer learning, trains its head on a laptop CPU, fast on a free CPU host. It's a CNN, so Grad-CAM heatmaps are simple (`target_layers=[model.conv_head]`). |
| **Zero-shot baseline** (no training) | **SigLIP 2** — `google/siglip2-base-patch16-224` | Apache 2.0 | base | Google's successor to CLIP, with better zero-shot classification. Classifies from text prompts alone, so it's our comparison model and a Day-1 fallback. |
| **Multi-object detection** (bonus) | **OWLv2** — `google/owlv2-base-patch16-ensemble` | Apache 2.0 | base | Open-vocabulary: finds boxes for "fighter jet", "tank", "warship" with **no box-labelled data**. Built into HF `transformers`. Each box is cropped, then classified by EfficientNet-B0. |
| **Explainability** (bonus) | **Grad-CAM** — `pytorch-grad-cam` | MIT | — | A heatmap showing which pixels drove the prediction. Also a debugging tool (it reveals whether the model looks at the sky or water instead of the object). |

### Strong alternatives (swap in if time or results justify)

| Role | Model | Licence | When to use |
|---|---|---|---|
| Zero-shot | **Meta Perception Encoder** — `PE-Core-B16-224` (timm/open_clip) | Apache 2.0 | Very strong on hard zero-shot benchmarks (ObjectNet, ImageNet-A). Test it against SigLIP 2 and keep the winner. |
| Main classifier | **DINOv2-small** + linear head | Apache 2.0 | Excellent frozen features when data is very small. Heatmaps on ViTs need more work. |
| Main classifier | **EfficientNetV2-S** (timm) | Apache 2.0 | Higher accuracy if a free Kaggle GPU is used for full fine-tuning. Slower on a CPU host. |
| Detection | **Florence-2-base** (Microsoft) | MIT | Open-vocabulary detection **plus captioning** in one ~0.23B model. The caption could strengthen the "explanation" (MUST 6). |
| Detection | **Grounding DINO (tiny)** | Apache 2.0 per official repo *(one source says GPL — verify before use)* | More accurate than OWLv2, but slower on CPU. |

### Deliberately avoided

| Model | Reason |
|---|---|
| YOLO-World / Ultralytics YOLO | GPL/AGPL licence, and standard YOLO needs box-labelled data we don't have |
| Grounding DINO 1.5 / 1.6 Pro | Paid API only, not free |
| DINOv3 | Custom restrictive licence |
| ConvNeXt | Harder to fine-tune on small datasets |
| Any cloud vision API (Google, AWS, OpenAI) | Costs money or needs a card |

---

## 2. Classes

| # | Class | Example sources |
|---|---|---|
| 1 | Fighter / military aircraft | Military/Civilian Vehicles, Aerospace Images |
| 2 | Helicopter | Military/Civilian Vehicles, Aerospace Images |
| 3 | Drone / UAV | Drone vs Bird |
| 4 | Military land vehicle / tank | Military/Civilian Vehicles, Military Vehicles |
| 5 | Naval vessel | Ships Image Dataset (Carrier, DDG, Submarine) |
| 6 | **Civilian / non-military** | Civilian classes from the same datasets (airliners, cars, cargo ships, birds) — hard negatives that sharpen boundaries such as fighter vs airliner and drone vs bird |

Anything else (a cat, a selfie) is handled by the **uncertainty / out-of-distribution gate** (§4), not by a class.

---

## 3. Data plan

1. **Base:** ASTRA starter pack. It's weak, so it's used as an extra test set plus a small amount of training data.
2. **Main data:** public Kaggle datasets, downloaded manually as zips into `data/raw/`.

   **Must download**

   | # | Kaggle dataset | Gives us |
   |---|---|---|
   | 1 | [Military/Civilian Vehicles Image Classification](https://www.kaggle.com/datasets/mexwell/militarycivilian-vehicles-image-classification) (mexwell) | Tanks, military trucks, military aircraft, military helicopters, civilian cars, civilian aircraft (~6,800 images) |
   | 2 | [Aerospace Images](https://www.kaggle.com/datasets/gatewayadam/aerospace-images) (gatewayadam) | Warplanes, helicopters, airliners, blimps, balloons |
   | 3 | [Drone vs Bird](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird) (muhammadsaoodsarwar) | Drones, plus birds as hard negatives |
   | 4 | [Ships Image Dataset](https://www.kaggle.com/datasets/vinayakshanawad/ships-dataset) (vinayakshanawad) | Carrier / DDG / Submarine (naval) plus cargo, cruise, sailboat, tug (civilian) |
   | 5 | [Military Vehicles](https://www.kaggle.com/datasets/amanrajbose/millitary-vechiles) (amanrajbose) | Tanks, APCs, IFVs — variety for the land class |
   | 6 | [Natural Images](https://www.kaggle.com/datasets/prasunroy/natural-images) (prasunroy) | Cats, dogs, flowers, fruit, people — **out-of-distribution test only, never trained on** |

   **Optional (only if a class is thin)**
   - [Military Helicopter Data Set](https://www.kaggle.com/datasets/crsuthikshnkumar/military-helicopter-data-set)
   - [Drone Image Classification](https://www.kaggle.com/datasets/balajikartheek/drone-type-classification)
   - [Military Aircraft Detection Dataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset) (large)
   - [The Warship Dataset](https://www.kaggle.com/datasets/queyrusi/the-warship-dataset)

   **Class mapping**

   | Class | Sources |
   |---|---|
   | Fighter / military aircraft | #1, #2 |
   | Helicopter | #1, #2 |
   | Drone / UAV | #3 |
   | Military land vehicle | #1, #5 |
   | Naval vessel | #4 (carrier, DDG, submarine) |
   | Civilian / non-military | #1 (cars, civilian aircraft), #2 (airliners, blimps), #3 (birds), #4 (civilian ships) |
3. **Balance:** cap at ~300–500 images per class.
4. **Deduplicate** with perceptual hashing (`imagehash`) **before** splitting, so near-duplicates can't leak between train and test and inflate accuracy.
5. **Split:** 70 / 15 / 15 (train / val / test), stratified, fixed seed. The test set is untouched until final evaluation.
6. **Licences:** check each dataset's licence and credit every source in the README.
7. **The dataset is not committed to git.** A `scripts/prepare_data.py` script and README instructions rebuild it.

---

## 4. Pipeline

```
USER
 │  image(s)
 ▼
FRONTEND (TBD)
 │  multipart upload
 ▼
FastAPI BACKEND
 ├─ Validation ── type check · PIL.verify() · size cap · pixel cap (decompression bomb)
 ├─ Preprocess ── EXIF rotation · → RGB · resize 224 · ImageNet normalise
 ├─ Model
 │    ├─ EfficientNet-B0 (fine-tuned)    ─┐
 │    └─ SigLIP 2 (zero-shot, prompts)   ─┴─ selectable via ?model=
 ├─ Post-process ── softmax · top-3 · uncertainty flag · explanation text
 ├─ Explainability ── Grad-CAM heatmap overlay (EfficientNet)
 ├─ Detect mode ── OWLv2 boxes → crop → classifier → labelled boxes
 └─ Storage ── SQLite (history) · thumbnails on disk
 │  JSON
 ▼
FRONTEND → USER
```

### Uncertainty / out-of-distribution gate
- **Low confidence:** top-1 below a threshold τ, **or** a top-1 vs top-2 gap below δ, gives an "Uncertain prediction" warning.
- τ and δ are **tuned on the validation set**, not guessed.
- **Temperature scaling** (fitted on val) so "87%" means roughly 87%.
- **Non-defence check:** SigLIP 2 compares the image against prompts like "a photo of an animal / person / food / landscape". If one of these wins, the result is "No defence object recognised".

### Explanation (MUST 6)
A template-built sentence from real outputs, for example:
> "**Helicopter** (87%). The model focused on the rotor and fuselage region (see heatmap). Second guess: Drone (8%)."

Stretch: use the Florence-2 caption for extra context.

---

## 5. API (frontend-agnostic)

| Method | Endpoint | Returns |
|---|---|---|
| `POST` | `/api/predict?model=effnet\|siglip` | class, confidence, top-3, uncertain flag, explanation, heatmap (base64) |
| `POST` | `/api/detect` | list of boxes, each with class, confidence, and an annotated image |
| `POST` | `/api/batch` | per-image results plus a CSV download |
| `GET` | `/api/history` | past predictions (gallery) |
| `DELETE` | `/api/history/{id}` | remove one entry |
| `GET` | `/api/metrics` | test-set metrics for both models plus confusion matrices |
| `GET` | `/api/models` | available models, classes, thresholds |
| `GET` | `/api/health` | status |

Errors return consistent JSON: `{ "error": "corrupt_image", "message": "..." }` with the right HTTP code (400 / 413 / 415 / 422).

---

## 6. Requirement coverage

| Tier | Requirement | Covered by |
|---|---|---|
| MUST | Accept image | `/api/predict` upload |
| MUST | Process with ML/CV model | EfficientNet-B0 / SigLIP 2 |
| MUST | Identify / classify | 5–6 classes |
| MUST | Show predicted class | response `class` |
| MUST | Show confidence | calibrated softmax |
| MUST | Simple explanation | template sentence + heatmap |
| SHOULD | Top-3 with confidences | `top3` |
| SHOULD | Preprocessing | EXIF, RGB, resize, normalise |
| SHOULD | Low-confidence warning | τ/δ gate, tuned on val |
| SHOULD | Bad-input handling | validation layer + tests |
| SHOULD | Model justification | README section |
| BONUS | Bounding boxes | OWLv2 + classifier |
| BONUS | Fine-tuning / transfer learning | EfficientNet-B0 on our data |
| BONUS | Explainable AI | Grad-CAM |
| BONUS | Model comparison | SigLIP 2 vs EfficientNet-B0, same test set |
| BONUS | Batch processing | `/api/batch` + CSV |
| BONUS | History / gallery | SQLite `/api/history` |
| BONUS | Performance metrics | accuracy, precision, recall, F1, confusion matrix |

---

## 7. Training and evaluation

**Training (`scripts/train.py`)**
- Stage 1: freeze the backbone, train the classifier head (runs on laptop CPU).
- Stage 2 (optional, Kaggle free GPU): unfreeze the last blocks and use a low learning rate.
- Augmentation: random resized crop, horizontal flip, colour jitter, slight rotation. **Background-aware:** random crops that reduce reliance on sky or water.
- Class-weighted loss if classes are imbalanced. Early stopping on val F1.
- Saves `models/effnet_b0.pt`, `class_names.json`, `calibration.json`.

**Evaluation (`scripts/evaluate.py`)**
- Both models are run on the **same held-out test set**.
- Outputs: accuracy, macro precision / recall / F1, per-class report, confusion matrix PNG, and reliability (calibration) plot.
- Out-of-distribution check: 30–50 non-defence images, measuring how often the gate correctly rejects them.
- Results go to `reports/metrics.json` and are served by `/api/metrics`.

---

## 8. Testing (`pytest`)

| Test | Expectation |
|---|---|
| Valid JPG / PNG / WebP | 200, prediction returned |
| Non-image (`.txt` renamed to `.jpg`) | 415 / clear error |
| Truncated / corrupt image | 400 `corrupt_image` |
| Huge file (> limit) | 413 |
| Enormous pixel count | rejected safely |
| RGBA / grayscale / palette / CMYK | converted, predicted |
| EXIF-rotated photo | correct orientation |
| Empty upload | 422 |
| Batch with one bad file | others succeed, bad one reported |
| Non-defence image (cat) | flagged uncertain / not recognised |

---

## 9. Repository structure

```
astra-vision/
├── app/
│   ├── main.py            # FastAPI app + routes
│   ├── config.py          # thresholds, paths, limits (env-driven)
│   ├── validation.py      # input checks
│   ├── preprocess.py
│   ├── models/
│   │   ├── effnet.py      # EfficientNet-B0 load + predict
│   │   ├── siglip.py      # SigLIP 2 zero-shot
│   │   └── detector.py    # OWLv2 + crop-classify
│   ├── explain.py         # Grad-CAM + explanation text
│   └── db.py              # SQLite history
├── scripts/
│   ├── prepare_data.py    # merge, dedupe, split
│   ├── train.py
│   └── evaluate.py
├── tests/
├── reports/               # metrics.json, confusion matrices
├── models/                # weights (git-ignored or Git LFS / HF Hub)
├── Dockerfile             # HF Spaces deploy
├── requirements.txt
├── .env.example
└── README.md
```

---

## 10. Stack and deployment

- **Python 3.12–3.14**, PyTorch (CPU), `timm`, `transformers`, `pytorch-grad-cam`, FastAPI, Uvicorn, Pillow, SQLite, scikit-learn, `imagehash`, pytest.
- **Training:** laptop CPU (head only) → Kaggle / Colab free GPU (full fine-tune, optional).
- **Hosting:** Hugging Face Spaces (free CPU, Docker). Weights hosted on the HF Hub.
- **Secrets:** none required (all models are local). `.env.example` is still provided for config.

---

## 11. Timeline

| When | Work |
|---|---|
| **29 Sep (today)** | Repo + FastAPI skeleton · validation + preprocessing · SigLIP 2 zero-shot working end to end · data download, dedupe, split · train EfficientNet-B0 head |
| **30 Sep** | Evaluate + compare · calibrate + tune thresholds · Grad-CAM · batch + history · OWLv2 detection · tests · deploy to HF Spaces · **frontend** |
| **1 Oct** | **No new features.** README, architecture diagram (Mermaid), AI-usage disclosure, record the 5–8 min video, final checklist, submit early |

**Priority if time runs short:** MUST → SHOULD → metrics + comparison → Grad-CAM → batch/history → detection.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overconfident on non-defence images | OOD gate + calibration + civilian class |
| Model learns backgrounds (sky = aircraft) | Grad-CAM inspection, crop augmentation. **Document as the debugging story in the video.** |
| Drone vs fixed-wing confusion | Confusion matrix, explain it honestly in Limitations |
| Duplicate images inflate accuracy | Perceptual-hash dedupe before splitting |
| OWLv2 slow on free CPU | Optional "Detect all" mode, smaller input size, clear loading state |
| Library incompatibility with Python 3.14 | Fall back to Python 3.12 |
| Running out of time | Strict priority order (§11) |

---

## 13. Video talking points (prepare now)

- **Why transfer learning:** small data, pretrained features, trainable on CPU.
- **Why EfficientNet-B0 over bigger models:** accuracy vs latency on free hosting.
- **Why SigLIP 2 as the baseline:** zero-shot, no training, fair comparison.
- **What confidence means (and doesn't):** calibration.
- **Debugging story:** the "cat → drone 91%" problem, or background shortcuts seen in Grad-CAM.
- **Limitations:** dataset size, domain shift (air-show photos vs real imagery), no satellite imagery.

---

## 14. Open decisions

- [x] Add the **Civilian / non-military** class — **yes**
- [x] Solo or team — **solo**
- [x] Dataset — starter pack is weak; **Kaggle datasets downloaded manually into `data/raw/`**
- [ ] **Frontend** choice — user has something different planned, TBD

---

## Sources

- [SigLIP 2 paper](https://arxiv.org/html/2502.14786v1)
- [Perception Encoder on HF (timm)](https://huggingface.co/timm/PE-Core-B-16) · [facebook/PE-Core-B16-224](https://huggingface.co/facebook/PE-Core-B16-224)
- [DINOv2 fine-tuning vs transfer learning](https://debuggercafe.com/dinov2-for-image-classification-fine-tuning-vs-transfer-learning/)
- [Image classification models 2026](https://labelyourdata.com/articles/image-classification-models)
- [Vision backbone selection in low-data regimes](https://arxiv.org/pdf/2410.08592)
- [Grounding DINO vs OWLv2](https://roboflow.com/compare/grounding-dino-vs-owlv2) · [Grounding DINO vs YOLO-World](https://playground.roboflow.com/models/compare/grounding-dino-vs-yolo-world)
- [Open-vocabulary detection overview](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr)
- [Best open-source CV models 2026](https://blog.roboflow.com/best-open-source-computer-vision-models/)
- [Open-vocabulary detectors on aerial imagery](https://arxiv.org/pdf/2601.22164)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) · [timm](https://github.com/rwightman/timm)
