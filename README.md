# 🔍 Sherlock — Real-Time Candidate Identification System

An AI-powered system that automatically identifies the interview candidate during live meetings using multi-signal Bayesian fusion. Built for the Sherlock internship challenge.

## The Problem

During live interviews on Google Meet, Teams, or Zoom, Sherlock's fraud detectors need to analyze the **candidate's** audio and video — not the interviewer or observers. But identifying who the candidate is can be surprisingly hard:

- Candidate joins as "MacBook Pro"
- Candidate uses a nickname ("Bill" instead of "William")
- Interviewer enters the wrong name
- Multiple interviewers + silent observers are present
- Candidate changes their display name mid-meeting

## My Approach

Instead of relying on a single heuristic (like name matching), I built a **multi-signal fusion engine** that combines 6 independent analyzers using **Bayesian probability updating**. Each analyzer produces a weak signal, and the engine combines them into a high-confidence identification.

### Key Insight
> No single signal is reliable enough on its own. But multiple weak signals, combined properly, produce confident identifications even in adversarial conditions.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Meeting Platform (Meet/Teams/Zoom)          │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
        ┌──────▼──────┐           ┌───────▼──────┐
        │ Audio/Video  │           │  Metadata    │
        │ Streams      │           │  Events      │
        └──────┬──────┘           └───────┬──────┘
               │                          │
┌──────────────▼──────────────────────────▼──────────────┐
│                 Signal Analyzers (6)                     │
│                                                          │
│  📝 Name Match    │ Fuzzy matching against candidate     │
│  🚪 Join Pattern  │ Timing & order analysis              │
│  🎙️ Speaking      │ Duration, turns, ratios              │
│  👁️ Behavioral    │ Webcam, screen share patterns        │
│  📄 Transcript    │ NLP on speech content                │
│  📅 Calendar      │ External metadata cross-reference    │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
        ┌──────▼──────────────────────────▼──────┐
        │        Bayesian Fusion Engine            │
        │   Log-likelihood weighted combination    │
        │   Posterior probability per participant   │
        └─────────────────┬───────────────────────┘
                          │
        ┌─────────────────▼───────────────────────┐
        │          Decision Engine                  │
        │   Confidence thresholds + explanations    │
        └─────────────────┬───────────────────────┘
                          │
        ┌─────────────────▼───────────────────────┐
        │        Real-Time Web Dashboard            │
        │   Live visualization & signal breakdown   │
        └─────────────────────────────────────────┘
```

## Signal Analyzers

| Signal | Weight | What It Analyzes |
|--------|--------|-----------------|
| **Name Match** | 30% | Fuzzy string matching (Levenshtein, Jaro-Winkler), nickname resolution, device name detection, email-to-name extraction |
| **Speaking Pattern** | 20% | Speaking ratio (candidates speak 35-65%), turn duration, turn count, recency |
| **Transcript** | 15% | Self-introduction detection, interview answer patterns (NLP regex), question vs answer ratio |
| **Calendar Metadata** | 13% | Calendar invite roles, known interviewer exclusion, company domain matching |
| **Join Pattern** | 12% | Join order, timing relative to schedule, session stability |
| **Behavioral** | 10% | Webcam on/off patterns, screen sharing, engagement consistency |

## Bayesian Fusion

The engine uses Bayesian updating:

1. **Prior**: Uniform distribution — each participant starts with equal probability
2. **Likelihood**: Each signal produces P(evidence | participant is candidate)
3. **Update**: Scores combined multiplicatively in log-space for numerical stability
4. **Posterior**: Normalized probabilities sum to 1

```python
# Effective score blends signal with neutral based on confidence
effective_score = signal.score * (confidence * weight) + 0.5 * (1 - confidence * weight)

# Log-likelihood update
log_posterior[participant] += log(effective_score)

# Normalize via log-sum-exp
posterior = softmax(log_posteriors)
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
git clone https://github.com/JillPatel28/sherlock-candidate-identification.git
cd sherlock-candidate-identification
pip install -r requirements.txt
```

### Run the Web Dashboard

```bash
python run_demo.py
```

Then open **http://localhost:5000** in your browser.

### Run CLI Demo

```bash
# Run a specific scenario
python run_demo.py cli standard
python run_demo.py cli device_name
python run_demo.py cli nickname

# Run all scenarios (evaluation mode)
python run_demo.py all
```

### Run Tests

```bash
pytest tests/ -v
```

## Demo Scenarios

| Scenario | Difficulty | Description |
|----------|-----------|-------------|
| Standard Interview | Easy | Normal 1-on-1, candidate name matches |
| Device Name | Medium | Candidate joins as "MacBook Pro" |
| Nickname | Medium | "Bill" instead of "William" |
| Wrong Name in System | Hard | System has "Michael" but candidate is "Michelle" |
| Panel + Observers | Hard | 3 interviewers, recording bot, late observer |
| Name Change | Medium | Joins as "Guest User 42", changes name later |
| Ambiguous (Stress Test) | Very Hard | Similar names, minimal metadata |

## Project Structure

```
sherlock-candidate-identification/
├── sherlock/                    # Core engine
│   ├── __init__.py
│   ├── engine.py               # Bayesian fusion engine
│   ├── models.py               # Data models
│   ├── simulator.py            # Meeting scenario simulator
│   └── signals/                # Signal analyzers
│       ├── name_matcher.py     # Fuzzy name matching
│       ├── join_pattern.py     # Join timing analysis
│       ├── speaking.py         # Speaking pattern analysis
│       ├── behavioral.py       # Webcam/screen share behavior
│       ├── transcript.py       # NLP transcript analysis
│       └── calendar.py         # Calendar metadata matching
├── web/                        # Web dashboard
│   ├── app.py                  # Flask server + REST API
│   └── templates/
│       └── dashboard.html      # Interactive dashboard UI
├── tests/
│   └── test_engine.py          # Unit + integration tests
├── run_demo.py                 # Main entry point
├── requirements.txt
├── ARCHITECTURE.md             # Detailed architecture doc
├── EVALUATION.md               # Testing & evaluation results
└── README.md                   # This file
```

## Assumptions

1. **Meeting platform provides participant metadata** — participant IDs, display names, join/leave events, webcam status, and speaker-attributed audio streams
2. **External metadata is available** — candidate name, email, interviewer names from calendar/ATS
3. **Audio streams are speaker-separated** — each participant has their own audio channel
4. **Transcript is available** — real-time speech-to-text with speaker attribution
5. **System runs server-side** — processing happens on Sherlock's infrastructure, not the meeting client

## Trade-offs & Design Decisions

| Decision | Trade-off |
|----------|-----------|
| Bayesian fusion over ML classifier | More interpretable, works without training data, but may be less accurate than a trained model |
| Regex-based transcript analysis | No external API dependency, fast, but less nuanced than LLM-based analysis |
| Weighted evidence combination | Principled probabilistic framework; weights start with heuristic defaults and adapt dynamically over time via online learning feedback |
| Simulated meeting data | Enables demo without real meetings, but may not capture all real-world edge cases |
| Uniform prior | No initial bias, but could use priors from historical data |

## What I'd Improve Next

1. **LLM-based transcript analysis** — Use GPT/Claude to understand conversational context more deeply
2. **Voice embeddings** — Compare voice signatures against known candidate recordings
3. **Face recognition** — Match webcam feed against candidate photo from application
4. **Multi-meeting correlation** — Use data from previous interview rounds
5. **Confidence calibration** — Ensure probability estimates are well-calibrated using historical data
6. **Real platform integration** — Connect to actual Meet/Teams/Zoom APIs

## License

MIT
