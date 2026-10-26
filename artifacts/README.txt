Put the training notebook's output here (the artifacts/ folder written by isic2024_v4_train.ipynb), so that this folder contains:

    meta.json                       (must say "version": "v4")
    isic_v4_lib.py
    backbones/<name>_fold<k>.pth    (e.g. eva02_small_fold0.pth ... swin_t_fold4.pth)
    oof_predictions.csv.gz          (used to calibrate the risk tiers)
    tabular_gbdt.joblib, fusion_gbdt.joblib, multiclass.joblib   (not used by the web app)

Then check it:   cd ml-service && python -m scripts.verify_artifacts --artifacts ../artifacts
