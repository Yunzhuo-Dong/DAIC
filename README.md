# Speech-based depression detection on E-DAIC

Predicting PHQ-8 depression severity from the speech of clinical interviews in the
[Extended Distress Analysis Interview Corpus (E-DAIC)](https://dcapswoz.ict.usc.edu/).

**Key result.** emotion2vec embeddings of the **participant's own speech**, isolated with speaker
diarization, reach **CCC 0.28 on the E-DAIC test set**. That is about 2.5× the best audio model of the
AVEC 2019 baseline (0.11). The same model on the whole recording drops to 0.16.

**No data is included.** E-DAIC is licensed for academic, non-commercial research only (request access at
<https://dcapswoz.ict.usc.edu/>).

---

## Data

- **Sessions.** 275 interviews with the virtual interviewer Ellie: 163 train / 56 dev / 56 test (AVEC 2019 split).
  The test set contains only AI-controlled interviews. Train and dev are mostly Wizard-of-Oz.
- **Label.** PHQ-8 total score (0–24), a self-report depression questionnaire.
- **Input used here.** The audio of the whole session and the ASR transcript. The transcript has no speaker labels.

## Method

1. **Participant speech.** Each recording contains the participant, Ellie (synthetic voice from a loudspeaker)
   and often an experimenter at the start.
    - We run [pyannote](https://github.com/pyannote/pyannote-audio) speaker diarization
      (`speaker-diarization-community-1`).
    - We then label the speakers with simple rules:
        - **Ellie**: the speaker who overlaps her scripted questions and is quieter or speaks less;
        - **experimenter**: the speaker active only in the first 2.5 min;
        - **participant**: the loudest remaining speaker.
    - Check: features of the selected speech predict the participant's gender with AUROC ≈ 1.0.
2. **Features**, one vector per session:
    - **eGeMAPS**: the provided openSMILE descriptors, statistics over participant frames;
    - **Praat**: F0, jitter, shimmer, HNR and intensity via parselmouth, plus speaking-rate and pause features;
    - **wav2vec2-large** (`facebook/wav2vec2-large-lv60`, frozen): mean over layers and over time;
    - **emotion2vec+ large** ([emotion2vec](https://github.com/ddlBoJack/emotion2vec), frozen): mean over
      20 s chunks.

    wav2vec2 and emotion2vec are computed on the **participant's speech only** or on the **whole recording**.
3. **Model.** Ridge regression (standardised features, alpha chosen by leave-one-out on the training data).
4. **Evaluation.** We report CCC (the AVEC metric):
    - 5 × 5-fold cross-validation on train + dev (219 people);
    - train → dev;
    - train → test. The test set was not used for training or tuning.

## Results

| Features (ridge) | Input | CV CCC (train+dev) | Dev CCC | **Test CCC [95 % CI]** | Test RMSE |
| --- | --- | --- | --- | --- | --- |
| eGeMAPS | participant | −0.012 | −0.046 | 0.104 [0.04, 0.18] | 6.34 |
| Praat | participant | 0.053 | −0.018 | 0.128 [0.05, 0.22] | 6.32 |
| wav2vec2-large | whole recording | 0.261 | 0.174 | 0.248 [0.13, 0.38] | 5.98 |
| wav2vec2-large | participant | 0.179 | 0.052 | 0.259 [0.11, 0.40] | 6.04 |
| emotion2vec+ large | whole recording | 0.325 | 0.388 | 0.164 [−0.06, 0.38] | 6.70 |
| **emotion2vec+ large** | **participant** | **0.346** | 0.369 | **0.280 [0.11, 0.44]** | **5.99** |

**AVEC 2019 baseline** ([Ringeval et al. 2019](https://arxiv.org/abs/1907.11510); one GRU per feature set):

| Model | Dev CCC | Test CCC |
| --- | --- | --- |
| eGeMAPS | 0.076 | – |
| MFCC | 0.198 | – |
| DS-VGG (best audio model) | 0.305 | 0.108 |
| Fusion of 10 audio + video models | 0.336 | 0.111 |

## Findings

1. **Isolating the participant matters more than the choice of representation.** On the whole recording,
   emotion2vec falls from CV 0.33 to test 0.16. On participant speech it keeps 0.28.
    - The difference comes from sessions where the participant speaks little, which are common in the AI test set.
    - In the third of test sessions with the lowest participant share, the whole-recording model gets CCC −0.24.
      There the session average is dominated by Ellie and background.
2. **Pre-trained speech representations beat hand-crafted acoustics.** eGeMAPS and Praat reach only
   CV CCC ≤ 0.05 and test CCC 0.10–0.13, about the level of the AVEC 2019 audio baseline. They predict
   values close to the mean (prediction SD ≈ 1.2 vs. true SD 6.4). wav2vec2 and emotion2vec reach CV CCC
   0.18–0.35 and test CCC 0.25–0.28.
3. **Emotion-specific pre-training helps.** On participant speech, emotion2vec beats wav2vec2 in both CV
   (0.35 vs. 0.18) and test (0.28 vs. 0.26).
4. **Better than the AVEC 2019 audio baseline.** Our best test CCC is 0.28, against 0.11 for the best AVEC
   audio model and for the AVEC audio-visual fusion.
5. **Limitations.**
    - Audio alone remains modest.
    - Participants who answer in a few words are hard for every model.
    - All models under-predict on the test set, whose mean PHQ-8 is 8.1 vs. 6.7 in train.

## Repository

```
01_index.py                  session table (split, labels, paths)
01b_diarize.py               pyannote diarization (GPU)
01c_select_participant.py    participant / Ellie / experimenter labelling
02_extract.py                feature extraction (eGeMAPS, Praat, wav2vec2, emotion2vec, ...)
03_train.py                  ridge models, 5x5 CV, dev / test predictions
04_evaluate.py               result tables with bootstrap CIs
configs/                     YAML configs (data.yaml is included by all others)
edaic/                       library code (data, features, models, metrics)
slurm/                       job scripts for the TU Dresden HPC (Capella GPU / Barnard CPU)
```

Example:

```bash
python 02_extract.py --config configs/extract_e2v_plus_large_full.yaml \
    --set segments.source=diarization extractor.name=e2v_plus_large
python 03_train.py --config configs/exp_feat_ridge.yaml --set features.0.name=e2v_plus_large name=e2v_part_ridge
python 04_evaluate.py --split test --runs $EDAIC_RUNS/e2v_part_ridge
```

## References

- F. Ringeval et al., "AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition," AVEC '19. [arXiv:1907.11510](https://arxiv.org/abs/1907.11510)
- Z. Ma et al., "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation," 2023. [GitHub](https://github.com/ddlBoJack/emotion2vec)
- A. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS 2020.
- H. Bredin et al., pyannote.audio. <https://github.com/pyannote/pyannote-audio>
