# Depression detection from clinical interviews on E-DAIC

Predicting PHQ-8 depression severity from the speech, transcripts and facial behaviour of clinical interviews in the
[Extended Distress Analysis Interview Corpus (E-DAIC)](https://dcapswoz.ict.usc.edu/).

**Key results** (E-DAIC test set; test data never used for training or tuning):

- **Audio.** emotion2vec embeddings of the **participant's own speech**, isolated with speaker diarization,
  reach **CCC 0.28**. That is about 2.5× the best audio model of the AVEC 2019 baseline (0.11).
- **Language model.** A local open-weight LLM (Qwen3.8-27B) rates the 8 PHQ-8 items from the participant's
  transcript without any fine-tuning. Its item scores, calibrated with ridge regression, reach **CCC 0.67**.
  That is significantly better than the best audio model (permutation p = 0.001).
- **Fine-tuning.** LoRA fine-tuning of the LLM on the 163 training interviews does not beat this
  zero-shot + calibration approach on the dev set (CCC 0.55 vs. 0.62).
- **Multimodal.** On top of the LLM, a leakage-free residual model on the participant's **voice** adds a small but
  consistent gain (CV +0.020, test **CCC 0.69**). It comes mainly from emotion2vec features, the change of the
  voice over the interview and slower answers. **Facial behaviour** (OpenFace) adds nothing.

**No data is included.** E-DAIC is licensed for academic, non-commercial research only (request access at
<https://dcapswoz.ict.usc.edu/>). All models run locally; no transcript was sent to an external service.

---

## Data

- **Sessions.** 275 interviews with the virtual interviewer Ellie: 163 train / 56 dev / 56 test (AVEC 2019 split).
  The test set contains only AI-controlled interviews. Train and dev are mostly Wizard-of-Oz.
- **Label.** PHQ-8 total score (0–24), a self-report depression questionnaire.
- **Input used here.** The audio of the whole session, the provided ASR transcript (no speaker labels) and the
  provided OpenFace 2.1 features (facial action units, head pose, gaze; no raw video).

## Method

### 1. Participant speech

Each recording contains the participant, Ellie (synthetic voice from a loudspeaker) and often an
experimenter at the start.

- We run [pyannote](https://github.com/pyannote/pyannote-audio) speaker diarization
  (`speaker-diarization-community-1`).
- We then label the speakers with simple rules:
    - **Ellie**: the speaker who overlaps her scripted questions and is quieter or speaks less;
    - **experimenter**: the speaker active only in the first 2.5 min;
    - **participant**: the loudest remaining speaker.
- Check: features of the selected speech predict the participant's gender with AUROC ≈ 1.0.

The participant's turns are used both for the audio features and for the participant transcript
(transcript rows that overlap participant turns, with Ellie's scripted phrases removed).

### 2. Audio models

One feature vector per session, followed by ridge regression (standardised features, alpha chosen by
leave-one-out on the training data):

- **eGeMAPS**: the provided openSMILE descriptors, statistics over participant frames;
- **Praat**: F0, jitter, shimmer, HNR and intensity via parselmouth, plus speaking-rate and pause features;
- **wav2vec2-large** (`facebook/wav2vec2-large-lv60`, frozen): mean over layers and over time;
- **emotion2vec+ large** ([emotion2vec](https://github.com/ddlBoJack/emotion2vec), frozen): mean over 20 s chunks.

wav2vec2 and emotion2vec are computed on the **participant's speech only** or on the **whole recording**.

### 3. LLM rating of the PHQ-8 items

The LLM plays the role of a clinician who reads the interview and fills in the questionnaire for the participant.

1. **Model.** [Qwen3.8-27B](https://huggingface.co/Qwen) (open weights, Apache-2.0), run in BF16 with
   Hugging Face `transformers` on one H100. Thinking mode is off and decoding is greedy (temperature 0),
   so every rating is deterministic.
2. **Input.** One prompt per session:
    - a system prompt: *"You are an experienced clinical psychologist … estimate the participant's answers to the
      PHQ-8 for the last two weeks. Each item is rated 0 = not at all … 3 = nearly every day. Base every rating on
      what the participant says; when an item is not discussed, give your best estimate from the overall
      impression. Answer with JSON only."*
    - the transcript (participant turns only, about 1–4 k tokens);
    - the 8 PHQ-8 items (anhedonia, depressed mood, sleep, fatigue, appetite, self-worth, concentration,
      psychomotor);
    - the required output: `{"items": [i1, …, i8], "evidence": "one short sentence"}`.
3. **Output.** The 8 item scores (0–3) are parsed from the JSON. All 275 answers were valid on the first try.
   The sum of the items is the **zero-shot** PHQ-8 prediction.
4. **Calibration.** The zero-shot scores are biased low (mean prediction 5.0 vs. true 6.6). We therefore use the
   8 item scores plus their sum as a 9-dimensional feature vector and fit the same ridge regression as for audio,
   on the training data only.
5. **Control.** The same prompt on the **full transcript** (Ellie and participant not separated) checks whether
   the LLM relies on the interviewer's follow-up questions.
6. **Fusion with audio.** LLM features + emotion2vec (early fusion: concatenation, one ridge model) and the mean of
   the two models' predictions (late fusion).

No LLM parameter is trained. Only the ridge calibration (9 weights) is fitted on the training data.

### 4. LLM fine-tuning (LoRA)

We also tested whether fine-tuning the LLM on the training labels beats the zero-shot model.

1. **Samples.** Each training sample is one interview and **one** PHQ-8 item: participant transcript + item text
   → the participant's true item score as a single digit (0–3). 163 training sessions × 8 items = 1304 samples.
   The loss is computed on the answer only.
2. **LoRA.** Rank 32, α 64, dropout 0.05, on all linear layers of the language model (attention, linear-attention
   and MLP projections; the vision encoder stays frozen). 234 M trainable parameters (0.86 % of 27 B), BF16.
3. **Training.** 1 epoch, effective batch size 16 (82 steps), AdamW, learning rate 2e-4 with 5 warm-up steps and
   cosine decay, gradient clipping at 1.0. About 32 min on one H100.
4. **Prediction.** One forward pass per (session, item). The probabilities of the tokens "0"–"3" give an expected
   item score (Σ k·p(k)); the 8 expected scores are summed. No text is generated, so every output is valid.
5. **Protocol.** Train on the training set and evaluate on dev. Only if dev beats the zero-shot + ridge model
   (dev CCC 0.624) would the model be retrained on train + dev and evaluated on test.
6. **Control.** The same per-item probability scoring with the original model (no training) separates the effect
   of fine-tuning from the effect of the scoring format.

### 5. Multimodal residual fusion

The text model stays the backbone; voice and face only try to correct what it misses:

```
final prediction = text prediction + λ · predicted residual,     residual = PHQ-8 − text prediction
```

1. **Text.** Ridge on the 9 zero-shot LLM features (section 3). Its training residuals are computed from
   **out-of-fold** predictions (inner 5-fold, repeated 3×); in-sample residuals would be too small.
2. **Residual model.** Ridge on standardised nonverbal features (median imputation; emotion2vec reduced to 32 PCA
   components). The ridge strength α and λ ∈ {0, 0.25, 0.5, 0.75, 1} are chosen by inner CV on the training data
   only. λ = 0 means text only, so the model can decide that a modality does not help.
3. **Audio features** (participant speech):
    - emotion2vec mean embedding;
    - its dynamics: spread of the segment embeddings, change between consecutive segments, and drift between
      the first and second half of the interview;
    - timing: speaking time, turn length, response latency after the interviewer, pauses, words per minute.
4. **Visual features** (OpenFace 2.1), split by speaker state using the diarization:
    - **speaking**: the participant's turns;
    - **silent**: the gaps between them, i.e. listening to the next question or pausing.

   For each state there are 35 statistics:
    - level and occurrence of 8 action units;
    - expressivity;
    - speed of facial, head and gaze movement;
    - head-pose and gaze variability;
    - smile, frown and lip-corner-depressor events per minute.

   Together with the speaking − silent contrasts this gives 80 features. A control uses the same statistics over
   all frames, without speaker states.
5. **Video quality control**, decided from QC data only, before any PHQ-8 result:
    - audio–video alignment is checked by lip–speech synchrony (participant speech vs. mouth opening);
    - 3 sessions with a clear offset were corrected;
    - 23 of 275 sessions (8 %) without lip–speech synchrony or with < 50 % tracked frames get no visual
      features and fall back to the text prediction.
6. **Experiments.**
    - E0: text only;
    - E1: + audio;
    - E2: + visual (speaker states);
    - E3: + audio + visual;
    - E4a / E4b: E2 / E3 with visual features pooled over all frames;
    - a **permutation control**: E3 with the nonverbal features shuffled across sessions.

   The model for the test set was chosen by CV, and the test set was evaluated once.

### 6. Evaluation

We report CCC, the AVEC metric:

- 5 × 5-fold cross-validation on train + dev (219 people);
- train → dev;
- train → test, with 95 % bootstrap confidence intervals and a paired permutation test.

## Results

### Audio (ridge)

| Features | Input | CV CCC (train+dev) | Dev CCC | **Test CCC [95 % CI]** | Test RMSE |
| --- | --- | --- | --- | --- | --- |
| eGeMAPS | participant | −0.012 | −0.046 | 0.104 [0.04, 0.18] | 6.34 |
| Praat | participant | 0.053 | −0.018 | 0.128 [0.05, 0.22] | 6.32 |
| wav2vec2-large | whole recording | 0.261 | 0.174 | 0.248 [0.13, 0.38] | 5.98 |
| wav2vec2-large | participant | 0.179 | 0.052 | 0.259 [0.11, 0.40] | 6.04 |
| emotion2vec+ large | whole recording | 0.325 | 0.388 | 0.164 [−0.06, 0.38] | 6.70 |
| **emotion2vec+ large** | **participant** | **0.346** | 0.369 | **0.280 [0.11, 0.44]** | **5.99** |

### LLM (Qwen3.8-27B)

| Model | Input | CV CCC (train+dev) | Dev CCC | **Test CCC [95 % CI]** | Test RMSE |
| --- | --- | --- | --- | --- | --- |
| Zero-shot (sum of items) | full transcript | 0.643* | – | 0.643 [0.49, 0.77] | 5.00 |
| Zero-shot (sum of items) | participant | 0.614* | – | 0.629 [0.48, 0.75] | 5.05 |
| **Zero-shot items + ridge** | **participant** | **0.633** | **0.624** | **0.666 [0.53, 0.77]** | **4.59** |
| + emotion2vec, early fusion | participant | 0.628 | 0.573 | 0.670 [0.51, 0.79] | 4.58 |
| + emotion2vec, late fusion | participant | – | – | 0.557 [0.44, 0.66] | 4.71 |

\* Zero-shot involves no training; the value is the CCC on all 219 train + dev sessions.

LLM items + ridge vs. emotion2vec (participant) on test: ΔCCC = +0.386, paired permutation p = 0.0012.

### LLM fine-tuning (dev set, model trained on train only)

| Model | Dev CCC | Dev MAE | Mean prediction (true 6.6) |
| --- | --- | --- | --- |
| Per-item probability scoring, no training (control) | 0.402 | 3.93 | 3.6 |
| Per-item LoRA fine-tuning, 1 epoch | 0.553 | 3.44 | 5.9 |
| **Zero-shot items + ridge (main model)** | **0.624** | – | – |

Fine-tuning did not reach the zero-shot + ridge model on dev, so it was not retrained or evaluated on test.
Per-item dev CCC after fine-tuning: interest 0.40, mood 0.50, sleep 0.48, energy 0.26, appetite 0.45,
self-worth 0.57, concentration 0.35, psychomotor 0.21.

### Multimodal residual fusion

| Experiment | CV CCC (train+dev) | Δ CV vs. text [SD over 5 repeats] | Dev CCC | **Test CCC [95 % CI]** | Test RMSE |
| --- | --- | --- | --- | --- | --- |
| E0: text (Qwen items + ridge) | 0.633 | – | 0.624 | 0.666 [0.53, 0.77] | 4.59 |
| **E1: + audio (chosen by CV)** | **0.652** | **+0.020 [0.008]** | 0.620 | **0.688 [0.55, 0.80]** | **4.51** |
| E2: + visual, speaker states | 0.628 | −0.004 [0.002] | 0.627 | 0.673 [0.54, 0.77] | 4.57 |
| E3: + audio + visual | 0.646 | +0.013 [0.008] | 0.610 | 0.702 [0.58, 0.80] | 4.44 |
| E4a: + visual, all frames | 0.630 | −0.002 [0.003] | 0.621 | 0.670 [0.55, 0.76] | 4.57 |
| E4b: + audio + visual, all frames | 0.642 | +0.010 [0.007] | 0.608 | 0.688 [0.56, 0.79] | 4.49 |
| Permutation control (E3 shuffled) | 0.628 | −0.005 [0.003] | 0.624 | 0.666 [0.53, 0.77] | 4.59 |

- E1 vs. E0 on test: ΔCCC = +0.022, paired permutation p = 0.50 (n = 56). The test difference has the same size
  as in CV but is not significant with 56 people.
- E3 is highest on test but was not chosen, because its CV score is lower than E1's.
- Largest residual-model weights in E1:
    - emotion2vec PCA components;
    - the drift of the voice between the first and second half of the interview (+);
    - the median and mean response latency (+), i.e. slower answers mean a higher PHQ-8 than the text predicts.

### Comparison with the literature (E-DAIC test set)

| Model | Input | Test CCC |
| --- | --- | --- |
| AVEC 2019 baseline, DS-VGG ([Ringeval et al. 2019](https://arxiv.org/abs/1907.11510)) | audio | 0.108 |
| AVEC 2019 baseline, fusion of 10 audio + video models | audio + video | 0.111 |
| MLlm-DR, LLaMA-3-8B zero-shot ([2025](https://arxiv.org/abs/2507.05591)) | text | 0.54 |
| MLlm-DR, LLaMA-3-8B fine-tuned with LoRA | text | 0.69 |
| QuestMF, question-wise fusion ([Mandal et al., CLPsych 2025](https://aclanthology.org/2025.clpsych-1.4/)), mean of 3 seeds | text / text + audio + video | 0.615 / 0.662 |
| MLlm-DR, multimodal | text + audio + video | 0.72 |
| **Ours: emotion2vec+ large, participant** | audio | **0.280** |
| **Ours: Qwen3.8-27B zero-shot + ridge** | text | **0.666** |
| **Ours: + audio residual (E1)** | text + audio | **0.688** |

The AVEC 2019 baseline reports dev CCC 0.305 for its best audio model and 0.336 for its fusion.

## Findings

1. **What the participant says carries far more signal than how it sounds.** Without any fine-tuning, the LLM
   rating reaches test CCC 0.67, against 0.28 for the best audio model. It also beats the zero-shot result of
   MLlm-DR (0.54) and comes close to their fine-tuned model (0.69).
2. **A linear calibration of the LLM is enough; fine-tuning does not help.** The raw item sums are correlated
   with PHQ-8 but too low. Ridge on the 9 item features removes the bias (test RMSE 5.05 → 4.59) and is very
   stable (CV SD 0.003). LoRA fine-tuning on 163 interviews mainly corrects the same bias (dev mean prediction
   3.6 → 5.9) but stays below the 9-parameter calibration on dev (0.55 vs. 0.62). The training loss stayed at
   0.3–0.5, so this is not a sign of memorisation: with this little data the fine-tuned model does not learn
   more than the zero-shot model already knows. The difference is within the uncertainty of a 56-person dev set.
3. **The voice adds a little on top of the LLM; the face does not.**
    - Simple fusion does not work: concatenating emotion2vec with the LLM features gives the same test CCC
      (0.670 vs. 0.666), and equal-weight late fusion drops to 0.56.
    - Residual fusion, where the voice only predicts what the text model misses, gives a small but consistent gain
      (CV +0.020, positive in every repeat; test +0.022). The permutation control shows no gain.
    - The voice adds *how* and *how fast* something is said: a voice that changes over the interview, and slower
      answers (cf. psychomotor slowing, PHQ-8 item 8).
    - Facial action units, head pose and gaze add nothing (CV −0.004). Separating speaking from silent periods does
      not change this.
    - This matches other E-DAIC studies, where video alone is weak or negative and multimodal gains are small.
4. **The LLM does not depend on the interviewer.** The full transcript, including Ellie's follow-up questions,
   is only 0.01 better on test than the participant's words alone, well within the confidence intervals.
5. **Isolating the participant matters for audio.** On the whole recording, emotion2vec falls from CV 0.33 to
   test 0.16; on participant speech it keeps 0.28. The difference comes from sessions where the participant
   speaks little, which are common in the AI test set (CCC −0.24 in the third of test sessions with the lowest
   participant share).
6. **Pre-trained speech representations beat hand-crafted acoustics.** eGeMAPS and Praat stay at the level of
   the AVEC 2019 audio baseline and predict values close to the mean. Among pre-trained models, emotion2vec
   beats wav2vec2 in CV (0.35 vs. 0.18) and on test (0.28 vs. 0.26).
7. **Limitations.**
    - The test set has only 56 people; confidence intervals are about ±0.12, so gains of +0.02 cannot be
      significant there. The CV results on 219 people are the main evidence.
    - The LLM rates the appetite item almost always 0, because the interview rarely touches on it.
    - Participants who answer in a few words are hard for every model.

## Repository

```
01_index.py                  session table (split, labels, paths)
01b_diarize.py               pyannote diarization (GPU)
01c_select_participant.py    participant / Ellie / experimenter labelling
02_extract.py                audio feature extraction (eGeMAPS, Praat, wav2vec2, emotion2vec, ...)
03_train.py                  ridge models, 5x5 CV, dev / test predictions
04_evaluate.py               result tables, bootstrap CIs, late fusion, permutation tests
11_llm_score.py              zero-shot PHQ-8 item rating with a local LLM
12_text_embed.py             transcript embeddings (Qwen3-Embedding; not used in the results)
13_llm_inspect.py            LLM ratings vs. true item scores, with the model's evidence sentence
14_finetune_llm.py           per-item LoRA fine-tuning of the LLM (stages: base / dev / final)
15_visual_features.py        OpenFace features by speaker state (speaking / silent) + video quality control
16_audio_dynamics.py         dynamics of the participant's emotion2vec embeddings over the interview
17_residual_fusion.py        text backbone + nonverbal residual, nested CV, experiments E0-E4 + permutation control
configs/                     YAML configs (data.yaml is included by all others; llm_qwen.yaml, ft_qwen.yaml,
                             multimodal.yaml)
edaic/                       library code (data, features, models, metrics)
slurm/                       job scripts for the TU Dresden HPC (Capella GPU / Barnard CPU)
```

Example:

```bash
# audio: emotion2vec on participant speech
python 02_extract.py --config configs/extract_e2v_plus_large_full.yaml \
    --set segments.source=diarization extractor.name=e2v_plus_large
python 03_train.py --config configs/exp_feat_ridge.yaml --set features.0.name=e2v_plus_large name=e2v_part_ridge

# LLM: zero-shot item ratings, then ridge calibration
python 11_llm_score.py --config configs/llm_qwen.yaml
python 03_train.py --config configs/exp_feat_ridge.yaml --set features.0.name=llm_qwen38_27b_part name=llm_part_ridge

# LLM fine-tuning: control, then train -> dev
python 14_finetune_llm.py --config configs/ft_qwen.yaml --stage base
python 14_finetune_llm.py --config configs/ft_qwen.yaml --stage dev

# multimodal residual fusion (CV + dev; add --with-test once at the end)
python 15_visual_features.py --config configs/multimodal.yaml --qc
python 15_visual_features.py --config configs/multimodal.yaml
python 16_audio_dynamics.py --config configs/multimodal.yaml
python 17_residual_fusion.py --config configs/multimodal.yaml

# test evaluation
python 04_evaluate.py --split test --runs $EDAIC_RUNS/e2v_part_ridge $EDAIC_RUNS/llm_part_ridge $EDAIC_RUNS/mm_residual
```

## References

- F. Ringeval et al., "AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition," AVEC '19. [arXiv:1907.11510](https://arxiv.org/abs/1907.11510)
- MLlm-DR: multimodal LLM for depression recognition on E-DAIC, 2025. [arXiv:2507.05591](https://arxiv.org/abs/2507.05591)
- Z. Ma et al., "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation," 2023. [GitHub](https://github.com/ddlBoJack/emotion2vec)
- A. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS 2020.
- H. Bredin et al., pyannote.audio. <https://github.com/pyannote/pyannote-audio>
- Qwen Team, Qwen3.8 model family. <https://huggingface.co/Qwen>
- A. Mandal, D. Atzil-Slonim, T. Solorio, I. Gurevych, "Enhancing Depression Detection via Question-wise Modality Fusion," CLPsych 2025. <https://aclanthology.org/2025.clpsych-1.4/>
- T. Baltrušaitis et al., "OpenFace 2.0: Facial Behavior Analysis Toolkit," FG 2018.
