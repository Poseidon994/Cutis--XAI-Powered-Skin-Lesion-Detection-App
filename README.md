# Cutis

A skin-lesion check built around the ISIC 2024 ensemble from `isic2024_v4_train.ipynb`. Same shape as Ocula: React frontend, Express API, FastAPI model service, Docker.

- Sign in as a **patient** or a **clinician**. Each gets different screens, different wording, and a different report.
- Upload one close-up photo. The image ensemble (4 backbones x up to 5 folds, 2 TTA views) rates it from *low* to *high concern*.
- **Three explanation methods run in parallel** on the same photo: Grad-CAM (Score-CAM fallback), LIME and SHAP.
- An **AI agent (Groq)** writes the explanation for the role: plain language and next steps for patients; scores, supporting evidence, differential, work-up, management and medication options for clinicians. Follow-up chat included.
- **Downloadable PDF report**, **scan history** per lesion (trend chart, included in the agent's context and in the report), account deletion.

```
browser ── nginx/Vite ──► Express API (:4000) ──► FastAPI model service (:8000)
 React        /api         auth, roles, SQLite,       image ensemble, Grad-CAM, LIME, SHAP
                           files, Groq agent, PDF     reads artifacts/ (read-only)
```

## 1. Read this first: what the app can and cannot use from your pipeline

The full pipeline (`isic2024_v4_submit.ipynb`) needs **3D-TBP measurements and the other lesions of the same patient** (intra-patient context features) to feed the tabular and fusion GBDTs. A single uploaded photo has neither. So the web app uses the **image branch only**: the fold networks, standardised with the stored `oof_standardisation_stats`, averaged exactly as the submission notebook does for a small test set (`isic_v4_lib.image_base_scores`).

- `tabular_gbdt.joblib`, `fusion_gbdt.joblib`, `multiclass.joblib` are **not loaded** by the app.
- The fused score's calibration cannot be reused, so the risk tiers are derived from `oof_predictions.csv.gz` for the image-only score `s_img_mean` (see section 5).
- The model was trained on 15 mm lesion crops from total-body photography. Phone or dermoscopy photos are a **domain shift** whose accuracy is unknown. The UI says so and the cropper steers users towards tight, lesion-centred framing.

## 2. Artifacts: check these before anything else

The service needs the **output of the final v4 training notebook**:

```
artifacts/
  meta.json                     "version": "v4"
  isic_v4_lib.py
  backbones/<name>_fold<k>.pth
  oof_predictions.csv.gz
```

> The Kaggle dataset in your screenshot (`frozen_stack.joblib`, `stage2.joblib`, `type_cascade.joblib`, `oof_predictions.csv`, `paper_style_table.csv`) has the layout of an **earlier run**. It has no `fusion_gbdt.joblib` and no `isic_v4_lib.py`, and the final `isic2024_v4_train.ipynb` never writes those frozen/stage2/type_cascade files, so it is not what `isic2024_v4_submit.ipynb` expects either. If that is what ends up in `artifacts/`, the service detects it and stops with an explanation instead of loading the wrong files.

```bash
cd ml-service
pip install -r requirements.txt            # plus torch + torchvision
python -m scripts.verify_artifacts --artifacts ../artifacts
# optional, the real integrity check (needs the competition data): scores 300 training lesions through the app's own code path and
# compares them with the out-of-fold scores stored in oof_predictions.csv.gz (expect Spearman > 0.8)
python -m scripts.verify_artifacts --artifacts ../artifacts --isic-dir /path/to/isic-2024-challenge --n 300
```

## 3. Run it

**Docker** (CPU by default):

```bash
cp .env.example .env        # fill in JWT_SECRET, ML_SERVICE_KEY, CLINICIAN_INVITE_CODE, GROQ_API_KEY
# put the notebook's artifacts/ output into ./artifacts  (or set ARTIFACTS_PATH)
docker compose up --build
# GPU: set TORCH_INDEX in .env to the CUDA wheel index for your driver, then
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
Open http://localhost:8080. The model service takes about a minute to load the networks; the app shows a banner until it is ready. Set `COOKIE_SECURE=1` once you serve over HTTPS.

**Without Docker**: `cp .env.example .env`, `npm install` in `server/` and `web/`, install the Python requirements plus torch, then `./dev.sh` (needs Node >= 22.13; the API uses the built-in `node:sqlite`, so there is no native module to compile).

Sizing: all 20 networks in fp32 need about 2 GB of RAM. `ML_MAX_FOLDS=2` loads 8 networks (about 0.8 GB) at a small accuracy cost.

## 4. Roles

| | Patient | Clinician |
|---|---|---|
| Sign-up | open | needs `CLINICIAN_INVITE_CODE` (disabled in production while it is empty) |
| Result page | tier in words, suggested timing, plain-language explanation, steps, warning signs, questions for the doctor | scores (z, percentile, observed cancer rate and enrichment, operating points), type distribution, per-backbone table, explanation metrics and agreement, differential, work-up, management and medication options, disposition rule |
| Hidden | raw scores, type probabilities, any named cancer type | nothing |
| Scan form | body site, name for the spot, symptoms, duration, notes | adds a patient reference (an ID, never sent to the AI writer) |
| History | own scans, concern level over time | own scans by patient reference and lesion, score over time |
| Report | action sheet | results sheet with evidence, options and history |

Scans are private to the account that created them; a clinician sees only their own analyses.

## 5. How the rating is built

1. The photo is centre-cropped to a square and resized to 128 px with bicubic interpolation, which is what `decode_crop` did in training. The browser cropper lets the user place the lesion first.
2. Every loaded fold network scores the crop (identity + horizontal flip). The malignant and suspicious logits are z-scored with that fold's stored mean and SD, folds are averaged per backbone, backbones are averaged: this is `s_img_mean`.
3. **Tiers** are cut at operating points computed from `oof_predictions.csv.gz` with the notebook's own `pick_threshold`: *low* below the 95%-sensitivity point, *some concern* up to the 90% point, *elevated* up to the 80% point, *high* above it. The ring around the photo shows the percentile among benign lesions.
4. The clinician view also shows the observed cancer rate among out-of-fold lesions scoring at least as high, against the base rate (0.098%).
5. The type distribution (Melanoma / BCC / SCC) is the type head's softmax and is **conditional on malignancy**; it is labelled that way everywhere.

Known approximation: the out-of-fold score came from one fold network per lesion, while the app averages up to five. The average has slightly less spread, so scores run marginally conservative. A "low" result never rules cancer out and the UI says so.

## 6. The three explanation methods

All three explain the same quantity, the mean z-scored malignant logit of the explanation networks (fold 0 of every backbone by default, `XAI_FOLDS`), and run concurrently (the UI starts all three at once; the server handles them in parallel threads).

| Method | What it does | Notes |
|---|---|---|
| **Grad-CAM** | Gradient-weighted activations of the last feature stage of each network, averaged over networks | Chosen because all four backbones expose a last spatial stage (ConvNeXt/EdgeNeXt stages, Swin block group, EVA02 patch tokens as a 16x16 grid). **Score-CAM** (gradient-free) is used automatically if gradients vanish, or always with `XAI_CAM_METHOD=scorecam`. |
| **LIME** | About 40 SLIC superpixels are switched off in random combinations (replaced by the surrounding-skin colour); a local linear model gives each region's push up or down | Reports local R2 so a poor surrogate is visible. |
| **SHAP** | `shap.GradientExplainer` (expected gradients) against plain-skin backgrounds; attributions add up to the shift from "plain skin" to this lesion | Reports the completeness gap. A direct implementation takes over if `shap` fails. |

LIME and SHAP use red for "raises the rating" and blue for "lowers it"; Grad-CAM is a single-direction attention map. The clinician view reports where the evidence lies, concordance between the three maps (top-20% overlap, chance level about 0.11), and the numbers above. On CPU, tune `XAI_LIME_SAMPLES` and `XAI_SHAP_SAMPLES`; on a GPU the defaults finish in seconds.

Verified by tests: a toy network whose score depends only on a dark top-left corner is explained by all three methods with at least 80% of the positive evidence in that corner and the right sign.

## 7. The AI agent

- **Input**: a structured JSON context (scan details, tier, plain-language comparisons for patients or the full numbers for clinicians, a summary of each explanation, concordance, earlier scans of the same lesion with a trend label). **Groq never receives the photo, the patient reference, or the user's name.** Free-text notes and the lesion name are sent, and the form warns about that.
- **Output**: JSON in a fixed schema per role, validated and normalised. Missing parts are filled from the template.
- **Guarantees that do not depend on the model**: suggested timing (patients) and disposition rule (clinicians) come from the tier; the disclaimer is fixed; if a patient report asserts the person does or does not have cancer, it is replaced by the template.
- **Fallback**: with no `GROQ_API_KEY`, a Groq outage or invalid output, the app uses deterministic template text and says so ("Standard text, no AI model used"). The PDF states which it is.
- **Chat**: grounded in the same context, short answers, role-specific rules (no diagnosis, no medicines or named cancer types for patients, urgent-care advice for alarming symptoms).
- Models are configurable (`GROQ_MODEL`, `GROQ_FALLBACK_MODELS`); check console.groq.com for current names, they are retired from time to time.
- Clinician management and medication content is general, guideline-level and conditional on histopathology, with no doses. It is written by a language model: treat it as a draft for a clinician to check, not as clinical truth.

## 8. Privacy and security notes

httpOnly session cookie (SameSite=Lax) plus a required `X-Requested-With` header on state-changing requests, bcrypt, rate limits on sign-in and agent endpoints, helmet headers, upload size and magic-byte checks, ownership checks on every scan and file, a shared secret between the API and model service (the model service is not exposed by compose). Users can delete their account and all data. It is a student research prototype: it has not had a security review, and it stores health images, so run it only where you are allowed to, over HTTPS, with backups and access controls you trust.

## 9. Tests and what was verified

```bash
cd ml-service && python -m pytest tests -q      # 14: API contract, score equals the training library's own computation, error paths, 3 explanations in parallel, localisation tests
cd server && npm test                           # 22: auth/roles, upload, parallel explanations, agent (stub Groq), safety, history, PDF, deletion
cd web && npm test                              # 10: both roles' pages rendered from real API payloads
```

All pass in my environment. Also run once: real FastAPI service + Express + built web app together, a full clinician flow including a PDF built from real overlays.

**Not verified** (I could not): your real weights and artifacts (tests use the real architectures with random weights, so wiring is proven but scores are meaningless), the live Groq API (stubbed, including rate-limit, outage and bad-output paths), Docker builds, GPU execution, and the UI in a real browser (rendered in jsdom only, never looked at on screen; the PDF layout was inspected visually). Run `verify_artifacts` with `--isic-dir` first, then click through both roles.

CPU timings from the 1-core, 4-network test machine, three explanations running together: Grad-CAM 7 s, SHAP (8 samples) 41 s, LIME (60 samples) 60 s; a prediction takes about 4 s. Expect much less on a multi-core CPU or GPU.

## 10. Layout

```
ml-service/  app/{main,artifacts,ensemble,calibration,imaging}.py  app/explain/{gradcam,lime_xai,shap_xai,common}.py
             app/vendor/isic_v4_lib.py (exact copy; the artifact's copy wins)  scripts/verify_artifacts.py  tests/
server/       src/{app,db,config,constants}.js  routes/{auth,scans}.js  services/{ml,agent,context,report,xai}.js  test/
web/          src/pages/*  src/components/{Lens,LensCropper,XaiPanel,AgentPanel,Chat,Bits,Shell}.jsx  src/styles.css
```

## 11. Troubleshooting

| Symptom | Cause |
|---|---|
| `/health` says `error: ... not in the v4 layout` | The artifacts folder is from an older run; see section 2. |
| Missing `backbones/...pth` | Copy the whole `backbones/` folder (weight saving must have been on). |
| Pickle or timm errors loading weights | Match `timm` and `torch` to `meta.json -> versions`. |
| Tiers look off, "approximate" banner | `oof_predictions.csv.gz` is missing or unreadable. |
| Sign-in works but you are logged out at once | Cookie is `Secure` on plain http: set `COOKIE_SECURE=0`. |
| Explanations take minutes | CPU only: lower `XAI_LIME_SAMPLES` / `XAI_SHAP_SAMPLES`, or use a GPU. |
| Report says "Standard text" | No `GROQ_API_KEY`, or Groq was unreachable; see the server log. |

## 12. Next steps worth doing

A batch endpoint for clinicians with a TBP export (metadata CSV + images), which would let the tabular and fusion models and patient-context features run as in the submission notebook; clinician-patient sharing with consent; a prospective check of phone-photo accuracy before anyone relies on the tiers.
