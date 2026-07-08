# Evaluation — Sherlock Candidate Identification System

## Testing Strategy

The system is evaluated using a combination of:
1. **Unit tests** for each signal analyzer
2. **Integration tests** for the full engine pipeline
3. **Scenario-based evaluation** across 7 representative cases
4. **Edge case testing** for failure modes

## Test Results

### Unit Tests

**31 tests — 31 passed (100%)**

| Test Category | Tests | Covers |
|--------------|-------|-----------|
| Name Matching | 8 tests | Exact match, partial match, no match, interviewer match, device names, nicknames, email extraction |
| Join Pattern | 3 tests | Candidate timing, early joiner, late joiner |
| Speaking | 2 tests | Candidate ratio, silent observer |
| Behavioral | 2 tests | Webcam-only signal, screen sharing |
| Transcript | 2 tests | Self-introduction, interviewer questions |
| Calendar | 2 tests | Candidate role, organizer role |
| Engine | 8 tests | Standard scenario, device name, panel, nickname, all scenarios, confidence increase, explanations, signal breakdown |
| Edge Cases | 3 tests | Empty participants, single participant, name change |
| Online Learning | 1 test | Weight update rule, normalization, persistence cleanup |

### Scenario Evaluation

Run all scenarios with `python run_demo.py all`:

| Scenario | Difficulty | Expected Candidate | Status | Confidence | Correct? |
|----------|-----------|-------------------|--------|-----------|----------|
| Standard Interview | Easy | Priya Sharma | Identified | 100% | ✅ |
| Device Name | Medium | MacBook Pro (Jordan) | Identified | 92% | ✅ |
| Nickname | Medium | Bill Roberts | Identified | 100% | ✅ |
| Wrong Name | Hard | Michelle Chen | Identified | 100% | ✅ |
| Panel + Observers | Hard | Aisha Patel | Identified | 96% | ✅ |
| Name Change | Medium | Carlos Rodriguez | Identified | 100% | ✅ |
| Ambiguous | Very Hard | S. Taylor | Likely | 73% | ✅ |

## Edge Cases Tested

### 1. Device Name ("MacBook Pro")
**Challenge**: Candidate joins with a device name, no name to match against.
**How it's handled**: 
- Name matcher assigns low confidence (device detected)
- Speaking and transcript analyzers compensate with behavioral evidence
- System correctly identifies based on speaking patterns and transcript content

### 2. Nickname ("Bill" vs "William")
**Challenge**: Candidate uses a common nickname not in the system.
**How it's handled**:
- Built-in nickname dictionary maps "Bill" → "William"
- Fuzzy matching still catches partial name overlap ("Roberts" matches exactly)
- Combined with behavioral signals for robust identification

### 3. Wrong Name in System
**Challenge**: System expects "Michael Chen" but candidate is "Michelle Chen".
**How it's handled**:
- Fuzzy matching still produces moderate similarity (shared last name, similar first name)
- Transcript analysis detects candidate introducing themselves
- Speaking patterns confirm interview-answer behavior
- System achieves identification despite the data error

### 4. Multiple Observers
**Challenge**: 6 participants — which ones are relevant?
**How it's handled**:
- Observers have no speaking segments → speaking analyzer excludes them
- Recording bots have no webcam → behavioral analyzer excludes them
- Calendar metadata identifies known interviewer roles
- Only the candidate shows the combined pattern of: speaking substantively + webcam on + not a known interviewer

### 5. Name Change Mid-Meeting
**Challenge**: Candidate starts as "Guest User 42" and changes name later.
**How it's handled**:
- Initial analysis relies on behavioral signals (speaking, webcam)
- When name changes, engine re-runs analysis and name match score jumps
- Confidence increases dynamically as expected

### 6. Ambiguous Names (Same Last Name)
**Challenge**: "Sam Taylor" and "Pat Taylor" — both have same last name.
**How it's handled**:
- Name match gives moderate scores to both
- Join order, speaking patterns, and transcript differentiate
- System correctly identifies based on who answers interview questions vs. who asks them

## Accuracy Metrics

### Overall Accuracy
- **7/7 scenarios** correctly identify the candidate (100% on test set)
- **31/31 unit tests** passing (100%)
- Mean final confidence: **94.4%** (across all 7 scenarios)
- Lowest final confidence: **73%** (Ambiguous stress test — the only "likely" rather than "identified")
- 6/7 scenarios reach full "identified" status (≥75%), 1 reaches "likely" (≥55%)

### Signal Contribution Analysis

| Signal | Average Impact | Most Useful In |
|--------|---------------|----------------|
| Name Match | High (when name is correct) | Standard, Nickname |
| Speaking Pattern | High (always informative) | Device Name, Ambiguous |
| Transcript | Medium-High (with data) | Wrong Name, Panel |
| Calendar Metadata | Medium (when available) | Panel + Observers |
| Join Pattern | Low-Medium | Panel, Standard |
| Behavioral | Low-Medium | Device Name, Observers |

### Confidence Calibration
Confidence scores are generally well-calibrated:
- Scores above 75% correspond to clearly correct identifications (6/7 scenarios)
- Scores between 55-75% indicate correct but uncertain identifications (Ambiguous scenario at 73%)
- Scores below 55% indicate genuinely ambiguous situations (none in current test set)

## Known Limitations

1. **Simulated data only**: All testing uses generated scenarios, not real meeting data. Real meetings have more variability and noise.

2. **No audio/video processing**: The system analyzes metadata about audio/video (speaking duration, webcam status) but doesn't process actual audio or video content.

3. **Regex-based NLP**: Transcript analysis uses pattern matching, not semantic understanding. An LLM would handle more nuanced conversational patterns.

4. **Feedback-based online learning**: While starting weights are set using heuristic defaults, they are dynamically adjusted and improved over time using our gradient-style reinforcement feedback loop when the user confirms candidate decisions.

5. **Cross-meeting learning**: The system persists learned weights across meeting sessions inside `learned_weights.json`, meaning it continuously learns from each interview round.

6. **English-only transcript patterns**: The regex patterns for transcript analysis are English-only.

7. **Static difficulty**: The system doesn't adapt its confidence thresholds based on meeting complexity.

## Robustness Analysis

### What the system handles well:
- ✅ Missing name information (falls back to behavioral signals)
- ✅ Wrong/misspelled names (fuzzy matching + transcript)
- ✅ Large meetings with observers (behavioral filtering)
- ✅ Name changes during meeting (dynamic re-analysis)
- ✅ Nicknames and abbreviations (nickname dictionary)
- ✅ Ambiguous situations (reports uncertainty instead of guessing)

### What could challenge the system:
- ⚠️ Candidate who doesn't speak (rare but possible in some interview formats)
- ⚠️ All participants with generic names and no external metadata
- ⚠️ Proxy interview (someone else speaking for the candidate)
- ⚠️ Non-English interviews (transcript patterns are English-specific)
- ⚠️ Very short meetings (< 1 minute of data)

## Performance

| Metric | Value |
|--------|-------|
| Analysis latency per update | < 25ms |
| Memory usage | < 50MB |
| Startup time | < 2 seconds |
| Max participants tested | 6 |
| Total test execution time | < 1 second (31 tests in 0.59s) |
| Weight adjustment latency | < 1ms |

## Online Learning Evaluation

The online learning weight update mechanism was tested via simulation:
1. **Convergence Test**: Running a standard scenario and confirming the candidate increases the weight of highly accurate detectors (e.g. `name_match` and `calendar_metadata`) while decreasing the weight of less predictive ones.
2. **Persistence Test**: Confirming a candidate correctly writes weights to `learned_weights.json`. Re-instantiating the engine loads the newly learned weights instantly.
3. **Safety Boundaries**: Submitting multiple identical feedback cycles verifies that weights remain within the `[0.05, 0.50]` safety boundary and do not overflow or cause division-by-zero errors.
4. **Accuracy Retention**: Running all 7 scenarios using learned weights from the Standard scenario shows that identification remains 100% accurate, demonstrating that the learning process keeps the engine stable and reliable.

## How to Reproduce

All results in this document can be independently verified:

```bash
# 1. Run the full test suite (31 tests, expects 100% pass)
pytest tests/ -v

# 2. Run all 7 scenarios and verify identification accuracy
python run_demo.py all

# 3. Launch the interactive web dashboard
python run_demo.py

# 4. Run a single scenario in CLI mode
python run_demo.py cli standard
python run_demo.py cli ambiguous
```
