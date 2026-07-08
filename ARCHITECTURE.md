# Architecture — Sherlock Candidate Identification System

## System Overview

The Sherlock Candidate Identification System is designed to answer one question in real-time: **"Which meeting participant is the interview candidate?"**

The system uses a multi-signal fusion architecture inspired by sensor fusion in autonomous vehicles and ensemble methods in machine learning. Rather than relying on any single signal (which would be brittle), it combines multiple independent evidence sources using Bayesian reasoning.

## Architecture Diagram

```
                    ┌───────────────────────────────┐
                    │     Meeting Platform API       │
                    │  (Google Meet / Teams / Zoom)  │
                    └──────┬───────────────┬────────┘
                           │               │
                    ┌──────▼───┐    ┌──────▼───────┐
                    │  Event   │    │  Media       │
                    │  Stream  │    │  Streams     │
                    │          │    │  (Audio/     │
                    │ • Join   │    │   Video)     │
                    │ • Leave  │    │              │
                    │ • Webcam │    │              │
                    │ • Screen │    │              │
                    │ • Name   │    │              │
                    └──────┬───┘    └──────┬───────┘
                           │               │
              ┌────────────▼───────────────▼─────────────┐
              │          Event Processor                   │
              │  Normalizes events into internal models    │
              │  Maintains participant state               │
              │  Updates speaking segments                 │
              └────────────┬───────────────┬─────────────┘
                           │               │
       ┌───────────────────▼───────────────▼──────────────────┐
       │              Signal Analyzers (Independent)           │
       │                                                       │
       │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
       │  │  Name Match  │  │ Join Pattern │  │  Speaking   │  │
       │  │  Analyzer    │  │ Analyzer     │  │  Analyzer   │  │
       │  │             │  │              │  │             │  │
       │  │ • Fuzzy     │  │ • Order      │  │ • Ratios   │  │
       │  │   matching  │  │ • Timing     │  │ • Duration │  │
       │  │ • Nicknames │  │ • Stability  │  │ • Turns    │  │
       │  │ • Devices   │  │ • Activity   │  │ • Recency  │  │
       │  │ • Email     │  │              │  │            │  │
       │  │ Weight: 30% │  │ Weight: 12%  │  │ Weight:20% │  │
       │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │
       │         │                │                 │          │
       │  ┌──────┴──────┐  ┌─────┴────────┐  ┌────┴───────┐  │
       │  │ Behavioral  │  │  Transcript  │  │  Calendar  │  │
       │  │ Analyzer    │  │  Analyzer    │  │  Analyzer  │  │
       │  │             │  │              │  │            │  │
       │  │ • Webcam    │  │ • Intro      │  │ • Roles   │  │
       │  │ • Screen    │  │   detection  │  │ • Known   │  │
       │  │   share     │  │ • Answer     │  │   people  │  │
       │  │ • Engage-   │  │   patterns   │  │ • Domain  │  │
       │  │   ment      │  │ • Question   │  │   match   │  │
       │  │             │  │   detection  │  │           │  │
       │  │ Weight: 10% │  │ Weight: 15%  │  │ Weight:13%│  │
       │  └──────┬──────┘  └──────┬───────┘  └────┬──────┘  │
       └─────────┼────────────────┼────────────────┼─────────┘
                 │                │                │
       ┌─────────▼────────────────▼────────────────▼─────────┐
       │            Bayesian Fusion Engine                     │
       │                                                       │
       │  For each participant:                                │
       │   1. Start with uniform prior: P(C_i) = 1/N          │
       │   2. For each signal s:                               │
       │      effective = score * (conf * weight)              │
       │                  + 0.5 * (1 - conf * weight)          │
       │      log_post[i] += log(effective)                    │
       │   3. Normalize: posterior = softmax(log_posteriors)    │
       │                                                       │
       │  Properties:                                          │
       │   • Low-confidence signals have minimal impact        │
       │   • Multiple weak signals compound multiplicatively   │
       │   • Probabilities always sum to 1                     │
       │   • Numerically stable via log-sum-exp                │
       └──────────────────────┬────────────────────────────────┘
                              │
       ┌──────────────────────▼────────────────────────────────┐
       │              Decision Engine                           │
       │                                                        │
       │  Thresholds:                                           │
       │   • ≥ 75%  → IDENTIFIED (high confidence)             │
       │   • ≥ 55%  → LIKELY (needs more evidence)             │
       │   • ≥ 35%  → ANALYZING (gathering data)               │
       │   • < 35%  → UNCERTAIN (ambiguous situation)           │
       │                                                        │
       │  Explanation Generator:                                │
       │   • Ranks signals by impact magnitude                  │
       │   • Produces human-readable reasoning chain            │
       │   • Shows which signals support/oppose identification  │
       └──────────────────────┬────────────────────────────────┘
                              │
       ┌──────────────────────▼────────────────────────────────┐
       │              Output Layer                              │
       │                                                        │
       │  REST API:                                             │
       │   POST /api/start         Start new simulation         │
       │   POST /api/next-phase    Advance to next phase        │
       │   GET  /api/scenarios     List available scenarios      │
       │   GET  /api/signal-breakdown/<id>  Signal details      │
       │   GET  /api/confidence-history     Score timeline       │
       │                                                        │
       │  WebSocket:                                            │
       │   Real-time updates pushed to connected dashboard      │
       │                                                        │
       │  Dashboard:                                            │
       │   • Live probability bars per participant              │
       │   • Signal breakdown panel with explanations           │
       │   • Phase timeline visualization                       │
       │   • Scenario selector with 7 test cases               │
       └──────────────────────────────────────────────────────┘
```

## Data Flow

### Real-Time Processing Pipeline

```
Meeting Event → Event Processor → Update Participant State
                                         │
                                    ┌────▼────┐
                                    │ Trigger │
                                    │ Analysis│
                                    └────┬────┘
                                         │
                              ┌──────────▼──────────┐
                              │ Run All 6 Analyzers  │
                              │ (parallel, O(N×6))   │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ Bayesian Fusion      │
                              │ O(N × S)             │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ Decision + Explain   │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ Emit Result to       │
                              │ Dashboard/Callbacks  │
                              └─────────────────────┘
```

### Latency Analysis

| Component | Latency | Notes |
|-----------|---------|-------|
| Event ingestion | < 1ms | Simple state updates |
| Name matcher | < 5ms | Fuzzy string ops |
| Join pattern | < 1ms | Simple comparisons |
| Speaking analyzer | < 1ms | Arithmetic on stats |
| Behavioral analyzer | < 1ms | Boolean logic |
| Transcript analyzer | < 10ms | Regex over transcript |
| Calendar analyzer | < 2ms | String comparisons |
| Bayesian fusion | < 1ms | Log-space arithmetic |
| **Total per update** | **< 25ms** | **Well within real-time** |

## Design Principles

### 1. Independence of Signals
Each analyzer operates independently and has no knowledge of other analyzers. This ensures:
- A bug in one analyzer doesn't corrupt others
- New analyzers can be added without modifying existing ones
- Analyzers can be tested in isolation

### 2. Graceful Degradation
When information is missing, analyzers return neutral scores (0.5) with low confidence. The fusion engine handles this by effectively ignoring low-confidence signals, ensuring the system works even with partial information.

### 3. Explainability
Every score comes with a human-readable explanation. The decision engine ranks explanations by impact, so users (and the demo) can understand **why** a particular participant was selected.

### 4. Continuous Updating
The system doesn't make a one-time decision. It continuously re-evaluates as new evidence arrives, allowing confidence to grow (or change) over time. This is critical for handling scenarios where the candidate starts with a generic name and changes it later.

## Online Learning & Weight Adaptability

To satisfy the requirement of continuously improving as more interview data becomes available, Sherlock features a **gradient-style feedback reinforcement loop** to adjust analyzer weights dynamically based on outcome feedback.

### Feedback Loop Pipeline
```
User Confirmation (Dashboard) 
            │
            ▼
      POST /api/feedback
            │
            ▼
   learn_from_feedback() (Engine) ──► Re-fuse using updated weights ──► Animate Dashboard
            │
            ▼
   learned_weights.json (Persistence)
```

### Weight Update Rule
When a user confirms a participant $C^*$ as the true candidate, the engine compares the scores generated by each analyzer against the confirmed truth:

1. **For the confirmed candidate ($C^*$)**: Signals that gave scores $> 0.5$ (correct support) are reinforced positively; signals with scores $< 0.5$ (incorrect opposition) are penalized.
2. **For all other participants ($P \neq C^*$)**: Signals that gave scores $< 0.5$ (correct opposition) are reinforced positively; signals with scores $> 0.5$ (incorrect support) are penalized.

Mathematically, the weight delta $\Delta w_i$ for signal $i$ is calculated as:
$$\Delta w_i = \eta \sum_{p \in \text{Participants}} (Score_{i,p} - 0.5) \cdot Direction_p \cdot Confidence_{i,p}$$
where:
- $\eta$ is the learning rate (default: $0.05$)
- $Direction_p = 1.0$ if $p == C^*$, else $-1.0$
- $Confidence_{i,p}$ is the analyzer's confidence for that participant's signal score.

The weights are then clamped to a reasonable operational boundary $[0.05, 0.50]$ (to prevent any single signal from completely dominating or collapsing to zero) and normalized to sum to exactly $1.0$. Updated weights are persisted in `learned_weights.json` and loaded automatically on subsequent engine initializations.

## Scaling Considerations

For production deployment at scale:

1. **Horizontal scaling**: Each meeting's identification is independent — can run on separate workers
2. **Signal computation**: All 6 analyzers can run in parallel (no dependencies between them)
3. **Event streaming**: Replace REST polling with Kafka/Redis streams for true real-time
4. **Model serving**: Transcript analysis could offload to GPU workers for LLM-based analysis
5. **State management**: Use Redis for participant state to enable stateless API servers
