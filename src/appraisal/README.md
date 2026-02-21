# Appraisal Training Notes

`appraisal` holds model and training primitives (stacked ensemble, temporal CV, conformal calibration, routing, population splitting).

## Retraining Flow

1. Retraining trigger event arrives on `retraining_triggers`.
2. Service loads recent training frame from PostgreSQL (`PostgresStore.fetch_training_frame`).
3. `train_segmented_models` runs by population and region/segment groups.
4. Artifacts are written to `MODEL_ARTIFACT_DIR` as `joblib` payloads.
5. `model_versions` table is updated with metrics (`mae`, `rmse`, `samples`).
