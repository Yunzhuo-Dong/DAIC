# Depression detection from clinical interviews on E-DAIC

Predicting PHQ-8 depression severity from the speech and transcripts of clinical interviews in the
[Extended Distress Analysis Interview Corpus (E-DAIC)](https://dcapswoz.ict.usc.edu/).

**Key results** (E-DAIC test set; test data never used for training or tuning):

- **Audio.** emotion2vec embeddings of the **participant's own speech**, isolated with speaker diarization,
  reach **CCC 0.28**. That is about 2.5× the best audio model of the AVEC 2019 baseline (0.11).
- **Language model.** A local open-weight LLM (Qwen3.8-27B) rates the 8 PHQ-8 items from the participant's
  transcript without any fine-tuning. Its item scores, calibrated with ridge regression, reach **CCC 0.67**.
  That is significantly better than the best audio model (permutation p = 0.001).

**No data is included.** E-DAIC is licensed for academic, non-commercial research only (request access at
<https://dcapswoz.ict.usc.edu/>). All models run locally; no transcript was sent to an external service.

---

## Data

- **Sessions.** 275 interviews with the virtual interviewer Ellie: 163 train / 56 dev / 56 test (AVEC 2019 split).
  The test set contains only AI-controlled interviews. Train and dev are mostly Wizard-of-Oz.
- **Label.** PHQ-8 total score (0–24), a self-report depression questionnaire.
- **Input used here.** The audio of the whole session and the provided ASR transcript. The transcript has no
  speaker labels.

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

### 4. Evaluation

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

### Comparison with the literature (E-DAIC test set)

| Model | Input | Test CCC |
| --- | --- | --- |
| AVEC 2019 baseline, DS-VGG ([Ringeval et al. 2019](https://arxiv.org/abs/1907.11510)) | audio | 0.108 |
| AVEC 2019 baseline, fusion of 10 audio + video models | audio + video | 0.111 |
| MLlm-DR, LLaMA-3-8B zero-shot ([2025](https://arxiv.org/abs/2507.05591)) | text | 0.54 |
| MLlm-DR, LLaMA-3-8B fine-tuned with LoRA | text | 0.69 |
| **Ours: emotion2vec+ large, participant** | audio | **0.280** |
| **Ours: Qwen3.8-27B zero-shot + ridge** | text | **0.666** |

The AVEC 2019 baseline reports dev CCC 0.305 for its best audio model and 0.336 for its fusion.

## Findings

1. **What the participant says carries far more signal than how it sounds.** Without any fine-tuning, the LLM
   rating reaches test CCC 0.67, against 0.28 for the best audio model. It also beats the zero-shot result of
   MLlm-DR (0.54) and comes close to their fine-tuned model (0.69).
2. **A linear calibration of the LLM is enough.** The raw item sums are correlated with PHQ-8 but too low.
   Ridge on the 9 item features removes the bias (test RMSE 5.05 → 4.59) and is very stable (CV SD 0.003).
3. **Audio adds nothing on top of the LLM.** Early fusion with emotion2vec gives the same test CCC (0.670 vs. 0.666)
   and a lower dev CCC; equal-weight late fusion drops to 0.56.
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
    - The test set has only 56 people; confidence intervals are about ±0.12.
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
configs/                     YAML configs (data.yaml is included by all others; llm_qwen.yaml for the LLM)
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

# test evaluation
python 04_evaluate.py --split test --runs $EDAIC_RUNS/e2v_part_ridge $EDAIC_RUNS/llm_part_ridge
```

## References

- F. Ringeval et al., "AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition," AVEC '19. [arXiv:1907.11510](https://arxiv.org/abs/1907.11510)
- MLlm-DR: multimodal LLM for depression recognition on E-DAIC, 2025. [arXiv:2507.05591](https://arxiv.org/abs/2507.05591)
- Z. Ma et al., "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation," 2023. [GitHub](https://github.com/ddlBoJack/emotion2vec)
- A. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS 2020.
- H. Bredin et al., pyannote.audio. <https://github.com/pyannote/pyannote-audio>
- Qwen Team, Qwen3.8 model family. <https://huggingface.co/Qwen>
