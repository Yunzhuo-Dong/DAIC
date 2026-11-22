# Speech-centred depression detection on E-DAIC

Predicting PHQ-8 depression severity from clinical interviews in the
[Extended Distress Analysis Interview Corpus (E-DAIC)](https://dcapswoz.ict.usc.edu/).
Speech is the core modality. This repository contains the complete pipeline:

1. extracting the participant's own speech with speaker diarization
2. extracting acoustic features and self-supervised speech representations
3. ridge baselines evaluated with repeated cross-validation
4. leakage and shortcut analyses

> **Status (Nov 2026):** audio-only track (RP1) complete on train + dev. Text and fusion (RP2) are next.
> The test set has **not** been evaluated yet; it will be run once, at the end.

**No data is included.** E-DAIC is licensed for academic, non-commercial research only. Request access at
<https://dcapswoz.ict.usc.edu/>. Raw data, features, diarization output and run folders live outside this repository.

---

## 1. Data

| | |
| --- | --- |
| Sessions | 275 participants: 163 train / 56 dev / 56 test (AVEC 2019 DDS split) |
| Interviews | Virtual interviewer "Ellie". IDs 300–492 are Wizard-of-Oz (human-controlled), IDs 600–718 are AI-controlled. **The test set is AI-only.** Train + dev contain 189 WoZ and 30 AI sessions |
| Per session | 16 kHz audio of the whole session, ASR transcript (start, end, text, confidence; **no speaker labels**), provided features (openSMILE eGeMAPS/MFCC, BoAW, Deep Spectrum VGG16 / DenseNet201, OpenFace) |
| Labels | PHQ-8 total (0–24), binary label PHQ-8 ≥ 10, per-item scores (train and dev only) |
| Label statistics | Train+dev PHQ-8 mean 6.64, SD 6.00, 15 % score 0. Depressed (≥ 10): train 37/163, dev 12/56, test 17/56 |

PHQ-8 is a validated **self-report screening** questionnaire, not a diagnostic gold standard. All results are
framed as *predicting self-reported depression severity*.

## 2. Participant speech: from transcripts to diarization

Each recording contains three voices: the participant (close-talk microphone, loud), Ellie (synthetic voice
from a loudspeaker, quiet), and in many sessions an experimenter setting up the equipment at the start.
The transcripts have no speaker column. ASR often merges Ellie's question and the answer into one row,
e.g. *"how are you doing today I'm doing good"*.

**Step 1: rule-based cleaning** (`edaic/data.py`, `configs/ellie_phrases.txt`, `00_ellie_check.py`).
This step:

- strips Ellie's scripted prompts, mined from n-grams that recur across many sessions;
- drops the setup talk before the interview starts (Ellie's intro, her first scripted question, or the first 40 s);
- drops rows containing setup keywords.

It was abandoned as the main method for two reasons. In some sessions the experimenter talks for more than
2 minutes. And Ellie's lines are often fused with the participant's answers.

**Step 2: speaker diarization** (`01b_diarize.py`).
This step runs [pyannote.audio](https://github.com/pyannote/pyannote-audio) 4.1,
`speaker-diarization-community-1`, in exclusive mode with 1–4 speakers. It takes about 25 s per session on
one H100, about 2 h for the whole corpus.

**Step 3: participant selection** (`01c_select_participant.py`).
The rules work without a GPU and are applied in order:

1. **Ellie**: a cluster that covers ≥ 50 % of the transcript rows matching her script, **and** is either
   ≥ 6 dB quieter than the loudest speaker or is not the longest speaker.
2. **Experimenter**: a cluster whose last turn ends before 150 s.
3. **Participant**: the loudest remaining cluster with ≥ 20 s of speech and ≥ 25 % of the longest remaining
   cluster. Clusters ≥ 60 s long, within 6 dB and with < 30 % Ellie overlap are merged in.

**Verification.**

| Check | Result |
| --- | --- |
| Sessions diarized | 275 / 275, 2–4 speakers (mean 2.8) |
| Ellie identified as her own cluster | 135 / 275. Her total speaking time is consistently about 1.5–2 min, as expected from a fixed script |
| Ellie-script rows falling inside participant turns | median 4 % |
| Participant speech | median 7.4 min (train 8.8, dev 7.5, **test 6.4**) |
| Gender predicted from participant-only features | AUROC 0.97–1.00 for every feature set (a mixed-in female synthetic voice would lower this) |
| Final rule vs. a simple "longest / loudest speaker" rule | differs in 7 sessions, each checked by hand |

Some AI-controlled sessions contain very little participant speech (about 1 min of one-word answers).
These sessions are kept, because brevity is itself a depression-related cue.

## 3. Features

All features are summarised per session, using only frames or segments inside the participant's turns,
unless the name says `_full` (whole recording) or `_others` / `_ellie` (non-participant speech only).

| Name | Source | Session vector |
| --- | --- | --- |
| `egemaps` | provided openSMILE eGeMAPS LLDs (10 ms) | mean, SD, p10, p50, p90 (115-d) |
| `praat` | parselmouth: F0, jitter, shimmer, HNR, intensity per turn, plus transcript timing | mean and SD over turns (28-d) |
| `boaw_egemaps` | provided bag-of-audio-words (100 bins, 0.1 s hop) | mean and SD (200-d) |
| `ds_vgg16`, `ds_densenet201` | provided Deep Spectrum (AVEC 2019 "DS-VGG" / "DS-DNet"), 1 s hop | mean and SD (8192-d / 3840-d) |
| `w2v2_base` | `facebook/wav2vec2-base`, frozen, 13 layers × 768 | per-layer time mean, duration-weighted over turns |
| `w2v2_large` | `facebook/wav2vec2-large-lv60`, frozen, 25 layers × 1024 | same; the main vector is the **mean over layers** |
| `participant_timing` | diarization turns + transcript | speaking time and share, turn count and length, **response latency**, own pauses, words per minute |
| `interview_structure` | diarization turns | recording length, Ellie / experimenter speech, number of non-participant turns |

## 4. Evaluation protocol

- **Heads.** Ridge regression for the PHQ-8 total (alpha chosen by leave-one-out), and a class-balanced
  ridge classifier for PHQ-8 ≥ 10. Both use median imputation and standardisation.
- **Main number: 5 × 5-fold cross-validation on train + dev (219 people).** A single 56-person dev set gives
  95 % bootstrap intervals about ±0.2 wide on CCC, so it is reported only as a secondary number.
- **Metrics.** CCC (the AVEC metric), RMSE and MAE for regression; AUROC, macro-F1 and balanced accuracy for
  classification. Bootstrap CIs over participants are in `04_evaluate.py`.
- **Test set.** Evaluated once per final model, at the end of the project (not done yet).

## 5. Results (audio only)

### 5.1 PHQ-8 regression and screening

| Features (participant speech) | CV CCC (5×5, n=219) | Dev CCC (n=56) | CV AUROC (PHQ ≥ 10) |
| --- | --- | --- | --- |
| eGeMAPS functionals | −0.012 ± 0.016 | −0.046 | 0.547 |
| Praat voice + timing | 0.053 ± 0.020 | −0.018 | 0.508 |
| BoAW-eGeMAPS | 0.020 ± 0.020 | −0.014 | 0.591 |
| Deep Spectrum VGG16 | 0.001 ± 0.019 | −0.195 | 0.508 |
| Deep Spectrum DenseNet201 | 0.023 ± 0.024 | −0.038 | 0.506 |
| participant timing only | 0.069 ± 0.016 | 0.012 | – |
| wav2vec2-base, layer mean | 0.094 ± 0.037 | 0.085 | 0.570 |
| wav2vec2-large, layer mean | 0.179 ± 0.027 | 0.052 | 0.595 |
| wav2vec2-large, mean of layers 12–18 | 0.212 ± 0.023 | 0.091 | 0.625 |
| **wav2vec2-large layer mean + timing** | **0.216 ± 0.027** | 0.102 | – |

Reference points from the AVEC 2019 baseline ([Ringeval et al. 2019](https://arxiv.org/abs/1907.11510),
GRU per feature set, learning rate tuned on dev):

| Model | Dev CCC | Test CCC |
| --- | --- | --- |
| eGeMAPS | 0.076 | – |
| DS-VGG (best single audio model) | 0.305 | 0.108 |
| Fusion of 10 audio-visual models | 0.336 | 0.111 |

### 5.2 Layer-wise probing

Probing every layer separately with ridge (participant speech only):

| Model | Best layer | Best CV CCC | Last layer CV CCC |
| --- | --- | --- | --- |
| wav2vec2-base | 6 / 12 | 0.218 | 0.052 |
| wav2vec2-large | 13 / 24 | 0.282 (dev 0.209) | 0.056 |

Depression-related information peaks in the **middle layers**; the last layers, specialised for the
pre-training objective, carry almost none. The best-layer numbers are selected post hoc and are therefore
optimistic. The pre-specified layer mean and the layer range 12–18 are the main results.

### 5.3 Participant-only vs. whole recording, and leakage

| wav2vec2-large layer mean | CV CCC | CV CCC, WoZ only | Gender AUROC* | WoZ-vs-AI AUROC |
| --- | --- | --- | --- | --- |
| participant turns | 0.179 | 0.122 | 1.000 | 0.849 |
| whole recording (`_full`) | **0.267** | **0.275** | 1.000 | 0.974 |
| all non-participant speech (`_others`) | 0.121 | 0.128 | 0.807 | 0.985 |
| Ellie clusters only (`_ellie`, n=99) | −0.022 | −0.028 | 0.700 | 0.958 |
| interview structure (no audio content) | −0.004 | – | – | – |
| WoZ/AI flag alone | −0.006 | – | – | – |

\* Gender of the **participant**, predicted from the given audio.

These results support the following interpretation:

- **The sub-corpus is not a confound.** The WoZ/AI flag alone does not predict PHQ-8, and the whole-recording
  advantage persists within WoZ sessions only.
- **No evidence of an interviewer-prompt shortcut in audio.** Ellie's voice alone predicts nothing (CCC −0.02).
  Burdisso et al. ([2024](https://arxiv.org/abs/2404.14463)) found such a shortcut for *text* on DAIC-WOZ.
  We did not find it in the audio channel, at least with linear models.
- **The "others" signal is participant leakage.** Non-participant audio still predicts the participant's
  gender (AUROC 0.81). Short backchannels and overlapping speech end up in other clusters.
- **Diarization is conservative.** The whole recording is better than participant turns most likely because
  diarization drops part of the participant's speech, not because of a shortcut. Recovering these segments
  (overlap-aware diarization, lower minimum duration) is a concrete next step.

## 6. Main findings so far

1. **Hand-crafted and provided acoustic features carry almost no linear signal** under a strict protocol:
   CV CCC ≤ 0.09. AVEC-style dev-only numbers are optimistic: their best audio model falls from 0.31 on dev
   to 0.11 on test.
2. **Self-supervised representations help, if the right layers are used.** wav2vec2-large middle layers
   reach CV CCC 0.21–0.28, versus 0.06 for the last layer.
3. **Timing behaviour is complementary** (+0.04 CCC when added to wav2vec2).
4. **Audio-only performance is modest** (CCC about 0.2, AUROC about 0.6), consistent with the literature.
   Strong E-DAIC results mostly come from text.
5. **The test set is AI-only and its participants speak less** (6.4 vs. 8.8 min in train). A drop from CV to
   test is expected and will be reported.

## 7. Next steps

- **RP2 – text and fusion.** Re-transcribe the participant turns with Whisper to get participant-only text,
  then compare text embeddings, open-weight LLM scoring (Qwen) and audio + text fusion.
- **Sequence model.** Attention pooling over participant turns (`attnpool`), 5 seeds.
- **Better participant recall.** Overlap-aware diarization; re-run the leakage table.
- **Per-item analysis.** Which PHQ-8 items (sleep, fatigue, psychomotor) audio predicts.
- **Final test evaluation.** Run once. Hypothesis: the whole-recording model degrades more on the AI-only
  test set than the participant-only model.

## 8. Repository layout

```
00_inspect.py               print real column names / formats of the downloaded files
00_ellie_check.py           check and extend the Ellie phrase list from the data
01_index.py                 sessions.csv: split, labels, paths, participant speech minutes
01b_diarize.py              pyannote diarization (GPU) -> $WS/edaic_clean/<id>/
01c_select_participant.py   rule-based participant / Ellie / experimenter labelling (--apply to write)
02_extract.py               cache one feature set: $EDAIC_FEAT/<name>/<id>.npz (resumable)
03_train.py                 ridge / ridge_clf / svr / attnpool; 5x5 CV on train+dev + dev predictions
04_evaluate.py              summary tables, bootstrap CIs, late fusion, permutation tests
05_llm.py                   reserved (RP2)
06_diagnose.py              sanity checks: gender AUROC, CV with gender-z / PCA, single-feature correlations
07_structure.py             interview_structure and participant_timing feature sets
08_leak_check.py            leakage analysis (section 5.3)
configs/                    YAML configs (data.yaml is included by all others), ellie_phrases.txt
edaic/                      data, features/, models, metrics, assemble, config, utils, llm
slurm/                      env files (Capella GPU / Barnard CPU), job scripts, ridge_suite.sh
```

## 9. Running the pipeline

This was developed on the TU Dresden ZIH HPC: Capella (H100) for GPU jobs, Barnard for CPU jobs.

The paths are set in `slurm/env.sh` and `slurm/barnard_env.sh`:

- `EDAIC_RAW`: the download
- `EDAIC_FEAT`: features
- `EDAIC_RUNS`: experiments
- `HF_HOME`: model cache

Diarization uses a separate environment (`~/diar_env`, Python 3.11, `pyannote.audio>=4.1`), so the main
environment's torch is not touched.

```bash
source slurm/env.sh
python 01_index.py --config configs/data.yaml
sbatch --export=ALL,CMD="python 01b_diarize.py --config configs/diarize.yaml" slurm/diar_job.sh
python 01c_select_participant.py --config configs/diarize.yaml --apply
sed -i 's/source: transcript /source: diarization /' configs/data.yaml && python 01_index.py
sbatch --export=ALL,CMD="python 02_extract.py --config configs/extract_w2v2_large.yaml" slurm/gpu_job.sh
python 07_structure.py --config configs/diarize.yaml
sbatch --export=ALL,CMD="bash slurm/ridge_suite.sh" slurm/cpu_job.sh
python 04_evaluate.py --runs $EDAIC_RUNS/*_ridge $EDAIC_RUNS/*_clf
python 08_leak_check.py --config configs/data.yaml
```

Any config value can be overridden from the command line, for example:

```bash
python 03_train.py --config configs/exp_feat_ridge.yaml \
  --set features.0.name=w2v2_large 'features.0.layer="mean"' features.1.name=participant_timing name=w2v_timing_ridge
```

## References

- F. Ringeval et al., "AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition," AVEC '19. [arXiv:1907.11510](https://arxiv.org/abs/1907.11510)
- S. Burdisso et al., "DAIC-WOZ: On the Validity of Using the Therapist's Prompts in Automatic Depression Detection from Clinical Interviews," 2024. [arXiv:2404.14463](https://arxiv.org/abs/2404.14463)
- M. Rodrigues Makiuchi et al., "Multimodal Fusion of BERT-CNN and Gated CNN Representations for Depression Detection," AVEC '19.
- A. Mandal et al., "Enhancing Depression Detection via Question-wise Modality Fusion," 2025. [arXiv:2503.20496](https://arxiv.org/abs/2503.20496)
- A. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS 2020.
- H. Bredin, pyannote.audio, <https://github.com/pyannote/pyannote-audio>
