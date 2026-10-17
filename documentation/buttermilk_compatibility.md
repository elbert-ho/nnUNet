# Buttermilk fork compatibility

This fork merges upstream nnU-Net through `bbf1222` (2026-10-16, version 2.8.1).
The pre-sync code is preserved on `backup/buttermilk-before-upstream-sync-20261017`.

The default trainer still builds its model through Buttermilk's `SingleDataExperiment`
and its loss through `pylot.experiment.util.eval_config`. It retains 2,000 epochs,
eight hypotheses per input, best-loss validation, disabled deep supervision and
compilation, and DDP with unused parameters allowed. Buttermilk and pylot must be
available in the training environment; they are imported when model/loss construction
is requested, so unrelated nnU-Net utilities can run without Buttermilk installed.

The original configuration path remains the default. Set `BUTTERMILK_NNUNET_CONFIG`
to select a different YAML file for training and inference. The custom predictor
`nnunetv2.inference.predict_from_raw_data_2.nnUNetPredictor` also accepts an explicit
`config_file` in `initialize_from_trained_model_folder`. Its model-building call now
uses upstream's plans/configuration-manager interface. It reads both legacy `.pth`
and upstream `.safetensors` checkpoints, including filename-extension fallback.
Use this custom predictor for eight-hypothesis exports. The `infer_multi.py` test-set
workflow and `nnUNetMultiRun.py` are retained.

Upstream now stores foreground sampling coordinates separately from case properties.
Run the updated `nnUNetv2_plan_and_preprocess` workflow for new data; for existing
preprocessed data, follow the sampling extraction CLI described in the upstream
preprocessed-data documentation. The upstream sampler is required for this workflow
and supports background/ignore-label sampling. The fork's previous foreground-only
sampler is retained as `DefaultPreprocessor._sample_foreground_locations_buttermilk`
for explicit use when reproducing legacy preprocessing.

The full upstream sync also changes dependency minimums; install this checkout in
your chosen environment to obtain those requirements. GPU training and real-data
inference require the external Buttermilk checkout, YAML configuration, and datasets.
