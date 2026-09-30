# The model, explained: what works, what doesn't, what's next

This walks through the [architecture diagram](../README.md#architecture) stage by stage. Numbers come from [`reports/metrics.json`](../reports/metrics.json).

## How a prediction flows

1. **Validation and preprocessing.** Bad files are rejected with a clear error. Good ones are fixed up (EXIF rotation, colour mode, transparency) and resized.
2. **Frozen encoder.** A pretrained network turns the image into a vector. Nothing in it is retrained. We use SigLIP 2 (default) or EfficientNet-B0 (comparison).
3. **Linear head.** A small logistic-regression layer (about 20 KB) maps the vector to the 5 classes. This is the only part we trained.
4. **Calibration.** A fitted temperature makes "90% confident" mean roughly 90% correct. The top-3 list and the "uncertain" flag come from here.
5. **Out-of-distribution (OOD) gate.** SigLIP 2 compares the image against defence and non-defence text prompts. If a non-defence concept wins, the answer is "no defence object recognised".
6. **Explanation.** A templated sentence explains the prediction.
7. **Detector (optional).** OWLv2 finds boxes from text queries. Each crop goes through the same classifier.
8. **History.** Results are stored in SQLite.

## What works

| Area | Evidence |
|---|---|
| Classification | SigLIP 2 + head: 99.2% accuracy, 100% top-3, at least 96.7% precision and recall on every class. Only two distinct held-out images were ever misclassified. |
| Calibration | ECE (expected calibration error) fell from 0.016 to 0.009 for the default model, so its stated confidence is trustworthy on this data. |
| Rejecting junk | The OOD gate rejects 10/10 unrelated images (maps, people, landscapes, drone-shot aerials). |
| Bad input | 20 tests cover empty, fake, truncated, oversized, tiny and decompression-bomb files. |
| Comparison | Three models on identical group-aware folds, so the comparison is fair. |
| Deployment | The Vercel demo runs the same SigLIP 2 model as ONNX and matches the PyTorch labels on 30/30 starter images. |

## What doesn't work well

| Problem | Detail | Cause |
|---|---|---|
| **The trained head adds nothing over zero-shot** | Both score 99.2%. Training only improved calibration slightly. | SigLIP 2 already knows these classes. The test set is too easy to separate them. |
| **EfficientNet confuses drones** | Drone precision 75% and recall 78%. It mixes drones with aircraft and helicopters. | An ImageNet CNN has weak features for small, oddly shaped airframes, and there are only 20 drone images. |
| **EfficientNet calibration got slightly worse** | ECE went from 0.032 to 0.035 after temperature scaling. | With so few images the fitted temperature is noisy. |
| **Two objects in one image** | A jet on a carrier was called a naval vessel. A ship carrying a helicopter was called a helicopter. | It is single-label classification, so it must pick one. |
| **Drawings and renders** | A line drawing of a flying-wing drone was called a military aircraft. | Training photos are almost all real photographs. |
| **Small, distant objects** | A far-off helicopter in a formation was labelled drone in detect mode. | The crop has too few pixels to classify. |
| **OOD gate is prompt-based** | It missed defence-context images such as interiors and ceremonies (2/23 rejected). It also wrongly rejects 3.4% of real defence images. | Text prompts describe topics, not "is this a military vehicle", so context leaks through. |
| **Vercel demo is reduced** | No per-object boxes. Uploads are capped near 4 MB. | That needs PyTorch, which does not fit in a serverless function. |
| **The numbers may be optimistic** | 117 images, many black-and-white WWII/Korean-war photos and 1860s engravings. | The dataset is small and skewed. Modern imagery is untested. |

## What failed during the build, and why

| Attempt | Outcome | Why it failed |
|---|---|---|
| Fine-tune the whole EfficientNet-B0 | Dropped | Only 117 clean images, so it would overfit. A frozen backbone plus linear head is the standard few-shot fix. |
| Kaggle datasets and a civilian class | Dropped | No time to download and clean them. The zero-shot OOD gate covers civilian objects instead. |
| Trust the starter data as-is | Reverted | 33 of 150 images were wrong (maps, drone-shot landscapes, ceremonies, interiors). They are excluded and listed in [`dataset/exclusions.csv`](../dataset/exclusions.csv). |
| Simple 70/15/15 split | Replaced | The test set would be about 18 images, and near-identical photos from one series would leak across the split. Group-aware cross-validation replaced it. |
| Running PyTorch on Windows | Moved to WSL | Smart App Control blocked PyTorch's DLLs. |
| PyTorch on Vercel | Replaced with ONNX | The runtime does not fit in a serverless function, so boxes are backend-only. |

## Solutions and next steps

1. **Harder, larger data.** Add modern aircraft, vehicle and ship sets, plus a *civilian* class (airliners, cargo ships, birds) as hard negatives. This is the single biggest gain, since it also shows whether the 99.2% holds up.
2. **Realistic test set.** Hold out modern, cluttered, weather-affected images that the model has never seen.
3. **Fine-tune on a free Kaggle GPU** once there are a few hundred images per class.
4. **Multi-label output** for mixed scenes, and always run detection when several objects are likely.
5. **Learned OOD detector** (for example Mahalanobis distance on the embeddings) instead of text prompts.
6. **Fine-grained types** (aircraft model, ship class) through a class hierarchy.
7. **Host the FastAPI backend** (Dockerfile is ready for Hugging Face Spaces) so the live demo can show boxes.

*This is a learning project on public imagery and is not validated for any real-world decision.*
