# Results

### Image: 3-way real / edited / AI (with 'uncertain') on data/sid/test

| metric | value |
|---|---|
| n | 696 |
| coverage | 0.6968 |
| accuracy_on_decided | 0.8887 |
| accuracy_counting_uncertain_as_wrong | 0.6193 |
| ai_auroc_real_vs_ai | 0.9936 |
| ai_auroc_real_vs_ai_UnivFD_official_baseline | 0.4432 |
| evaluated_on | held-out eval half (fusion trained on calibration half) |
| test_set | data/sid/test |

### 3-way fusion (trained on calibration half)

| metric | value |
|---|---|
| calib_n | 1583 |
| eval_n | 1413 |
| tau | 0.69 |
| tamper_flag | 0.901 |
| calib_accuracy_all | 0.8073 |
| calib_coverage_at_tau | 0.698 |
| calib_accuracy_on_decided | 0.9014 |

Recall per class: {"real": 0.5351, "edited": 0.4076, "ai_generated": 0.9217}

Confusion (truth -> predicted): {"real": {"ai_generated": 4, "uncertain": 86, "real": 122, "edited": 16}, "edited": {"uncertain": 109, "real": 22, "edited": 97, "ai_generated": 10}, "ai_generated": {"ai_generated": 212, "uncertain": 16, "edited": 1, "real": 1}}

Edited recall by edit kind: {"sid_tampered": 0.4076}

### Image: OpenFake test + our synthetic edits

| metric | value |
|---|---|
| n | 717 |
| coverage | 0.7001 |
| accuracy_on_decided | 0.9143 |
| accuracy_counting_uncertain_as_wrong | 0.6402 |
| ai_auroc_real_vs_ai | 0.9796 |
| ai_auroc_real_vs_ai_UnivFD_official_baseline | 0.4743 |
| evaluated_on | held-out eval half (fusion trained on calibration half) |
| test_set | data/images/test |

Recall per class: {"real": 0.524, "edited": 0.5758, "ai_generated": 0.8263}

### AI-image AUROC per generator

| generator | n | AUROC |
|---|---|---|
| z-image-turbo | 60 | 0.9704 |
| illustrious | 44 | 0.9994 |
| flux.2-klein-9b | 36 | 0.9756 |
| midjourney-7 | 20 | 0.9644 |
| gpt-image-1.5 | 18 | 0.9936 |
| veo-3 | 14 | 0.9846 |
| wan-video-2.5 | 9 | 0.9853 |
| ernie-image-turbo | 7 | 0.9794 |
| recraft-v3 | 5 | 0.876 |

### Image AI detector: clean vs daily-life degraded

| set | AUROC | acc | ECE |
|---|---|---|---|
| train | 0.9566425078491846 | 0.8859024044672251 | 0.05704158555782698 |
| img_test | 0.9799226907630523 | 0.9103206412825652 | 0.03676283916070434 |
| img_test_clean_only | 0.9796907630522088 | 0.9098196392785571 | 0.03980751066320074 |
| img_test_degraded_only | 0.9803935742971888 | 0.9108216432865731 | 0.033718167658207926 |
| sid_test | 0.9920640000000001 | 0.961 | 0.07602317498611215 |

### Edit localizer (patch level) - synthetic_openfake

| metric | value |
|---|---|
| patch_auroc | 0.8754908154256765 |
| image_auroc_edited_vs_clean | 0.8945258947436331 |
| mean_patch_iou_on_edited | 0.23356160521507263 |
| n_test_edited | 498 |
| n_test_clean | 498 |


### Edit localizer (patch level) - sid_set

| metric | value |
|---|---|
| patch_auroc | 0.8508621703146169 |
| image_auroc_edited_vs_clean | 0.8718269999999999 |
| mean_patch_iou_on_edited | 0.2835037410259247 |
| n_test_edited | 500 |
| n_test_clean | 1000 |

### Audio - test

| metric | value |
|---|---|
| n | 217 |
| auroc | 1.0 |
| acc | 0.9954 |
| auroc_degraded | 0.9995 |

### Audio - test_unseen

| metric | value |
|---|---|
| n | 231 |
| auroc | 0.9782 |
| acc | 0.8831 |
| auroc_degraded | 0.9816 |

### Text

| metric | value |
|---|---|
| n | 800 |
| auroc | 0.8826 |
| acc | 0.855 |
| auroc_clean | 0.975 |
| auroc_attacked | 0.7574 |

Per domain: {"abstracts": 0.6679, "books": null}

Baseline roberta_openai_gpt2_detector (openai-community/roberta-base-openai-detector): AUROC 0.8143, acc 0.6975 

Baseline roberta_chatgpt_detector_hc3 (Hello-SimpleAI/chatgpt-detector-roberta): AUROC 0.6336, acc 0.6388 


### Image AI detection: ours vs official UnivFD weights (same CLIP backbone)

| model | AUROC real vs AI |
|---|---|
| UnivFD official head (ProGAN-trained) | 0.4432 |
| Ours (OpenFake-trained, daily-life augmented) | 0.9936 |


### Explanations

LLM explanations passing the faithfulness check: 0.0833 (n=24)

**f66cd0ebebcca9c8.png** (truth: real)

```
Verdict: UNCERTAIN (the evidence is weak or conflicting), confidence 62%.
Main evidence:
- The learned detector gives a 27% probability that the whole image is AI-generated.
- The edit localizer finds no clearly altered region (score 0.58).
```

**5f2f7b01e1e9c05e.png** (truth: real)

```
Verdict: UNCERTAIN (the evidence is weak or conflicting), confidence 52%.
Main evidence:
- The learned detector gives a 8% probability that the whole image is AI-generated.
- The edit localizer finds no clearly altered region (score 0.83).
```

**tampered_03886.png** (truth: edited)

```
Verdict: likely AI-GENERATED, confidence 100%.
Main evidence:
- The learned detector gives a 92% probability that the whole image is AI-generated.
Evidence pointing the other way:
- The edit localizer finds no clearly altered region (score 0.63).
```

**tampered_03079.png** (truth: edited)

```
Verdict: UNCERTAIN (the evidence is weak or conflicting), confidence 59%.
Main evidence:
- The learned detector gives a 8% probability that the whole image is AI-generated.
- The edit localizer finds a region that looks pasted or altered, strongest in the top-left of the image (score 0.95).
```

**full_synthetic_006578.png** (truth: ai_generated)

```
Verdict: likely AI-GENERATED, confidence 100%.
Main evidence:
- The learned detector gives a 100% probability that the whole image is AI-generated.
Evidence pointing the other way:
- The edit localizer finds no clearly altered region (score 0.52).
```

**full_synthetic_009987.png** (truth: ai_generated)

```
Verdict: likely AI-GENERATED, confidence 85%.
Main evidence:
- The learned detector gives a 86% probability that the whole image is AI-generated.
Evidence pointing the other way:
- The edit localizer finds no clearly altered region (score 0.85).
```
