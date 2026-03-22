"""
agents/ — Specialist ML model inference wrappers.

Each agent wraps one trained model and returns a structured dict
for consumption by the LangGraph triage pipeline.

Available agents:
  cough_agent   : LightCoughCNN on COUGHVID mel spectrograms
                  -> Healthy / Symptomatic (binary cough classification)

  lung_agent    : MultiTaskEfficientNet on Resp-229K mel spectrograms
                  -> Disease head: Normal / COPD / Pneumonia / Asthma / Heart_Failure
                  -> Sound head:   Normal / Crackle / Wheeze / Both

  symptom_agent : XGBoost on COUGHVID metadata (8 tabular features)
                  -> Healthy / Symptomatic
"""