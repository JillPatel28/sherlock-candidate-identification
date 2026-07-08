"""
Bayesian Fusion Engine

The core of Sherlock's candidate identification system. Combines scores
from all signal analyzers using weighted Bayesian updating to produce
a unified probability for each participant being the candidate.

The key insight is that no single signal is reliable enough on its own,
but multiple weak signals combined can produce high-confidence identification.
"""

import time
import logging
from typing import Optional

import numpy as np

from sherlock.models import (
    Participant, MeetingContext, SignalScore, CandidateAssessment,
    IdentificationResult, TranscriptEntry
)
from sherlock.signals.name_matcher import NameMatchAnalyzer
from sherlock.signals.join_pattern import JoinPatternAnalyzer
from sherlock.signals.speaking import SpeakingAnalyzer
from sherlock.signals.behavioral import BehavioralAnalyzer
from sherlock.signals.transcript import TranscriptAnalyzer
from sherlock.signals.calendar import CalendarAnalyzer

logger = logging.getLogger(__name__)


# Confidence thresholds for identification decisions
CONFIDENCE_IDENTIFIED = 0.75     # above this, we're confident
CONFIDENCE_LIKELY = 0.55         # above this, we have a likely candidate
CONFIDENCE_UNCERTAIN = 0.35      # below this, we're uncertain


class CandidateIdentificationEngine:
    """
    Multi-signal Bayesian fusion engine for candidate identification.
    
    Architecture:
    1. Each signal analyzer independently scores every participant
    2. Scores are treated as likelihood ratios in Bayesian framework
    3. Prior is uniform across participants (no initial bias)
    4. Posterior probabilities updated with each new piece of evidence
    5. Explanations generated for the top-ranked participant
    
    The engine runs continuously, updating with every new event.
    """
    
    def __init__(self, context: MeetingContext):
        self.context = context
        self.participants = {}  # participant_id -> Participant
        self.transcript_entries = []
        self.event_history = []
        self.result_history = []
        
        # Initialize signal analyzers
        self.analyzers = {
            "name_match": NameMatchAnalyzer(),
            "join_pattern": JoinPatternAnalyzer(),
            "speaking_pattern": SpeakingAnalyzer(),
            "behavioral": BehavioralAnalyzer(),
            "transcript": TranscriptAnalyzer(),
            "calendar_metadata": CalendarAnalyzer(),
        }
        
        # Signal weights (learned from importance, can be tuned)
        self.signal_weights = {
            "name_match": 0.30,
            "join_pattern": 0.12,
            "speaking_pattern": 0.20,
            "behavioral": 0.10,
            "transcript": 0.15,
            "calendar_metadata": 0.13,
        }
        
        # Track the current identification state
        self._current_result = None
        self._update_count = 0
        self._callbacks = []
    
    def register_callback(self, callback):
        """Register a callback to be called on every identification update."""
        self._callbacks.append(callback)
    
    def add_participant(self, participant: Participant):
        """Add or update a participant."""
        self.participants[participant.participant_id] = participant
        logger.info(f"Participant added/updated: {participant.participant_id} ({participant.display_name})")
    
    def remove_participant(self, participant_id: str):
        """Mark a participant as inactive (left the meeting)."""
        if participant_id in self.participants:
            self.participants[participant_id].is_active = False
            self.participants[participant_id].leave_time = time.time()
    
    def add_event(self, event):
        """Process a new meeting event."""
        self.event_history.append(event)
        
        pid = event.participant_id
        if pid in self.participants:
            p = self.participants[pid]
            p.events.append(event)
            
            # Update participant state based on event
            from sherlock.models import EventType
            
            if event.event_type == EventType.WEBCAM_ON:
                p.webcam_on = True
            elif event.event_type == EventType.WEBCAM_OFF:
                p.webcam_on = False
            elif event.event_type == EventType.SCREEN_SHARE_START:
                p.is_screen_sharing = True
            elif event.event_type == EventType.SCREEN_SHARE_STOP:
                p.is_screen_sharing = False
            elif event.event_type == EventType.NAME_CHANGE:
                new_name = event.metadata.get("new_name", "")
                if new_name:
                    p.display_name = new_name
                    p.name_history.append(new_name)
            elif event.event_type == EventType.LEAVE:
                p.is_active = False
                p.leave_time = event.timestamp
            elif event.event_type == EventType.JOIN:
                p.is_active = True
                if p.join_time is None:
                    p.join_time = event.timestamp
    
    def add_speaking_segment(self, segment):
        """Add a speaking segment to the relevant participant."""
        pid = segment.participant_id
        if pid in self.participants:
            self.participants[pid].speaking_segments.append(segment)
            self.participants[pid].update_speaking_stats()
    
    def add_transcript_entry(self, entry: TranscriptEntry):
        """Add a transcript entry."""
        self.transcript_entries.append(entry)
    
    def identify(self) -> IdentificationResult:
        """
        Run the full identification pipeline.
        
        1. Collect scores from all analyzers for all participants
        2. Apply Bayesian fusion
        3. Generate explanations
        4. Return result with confidence
        """
        self._update_count += 1
        
        if not self.participants:
            return IdentificationResult(
                timestamp=time.time(),
                status="no_participants",
                explanation="No participants have joined the meeting yet"
            )
        
        participant_list = list(self.participants.values())
        
        # Update transcript analyzer with latest entries
        self.analyzers["transcript"].set_transcript(self.transcript_entries)
        
        # Step 1: Collect all signal scores
        all_scores = {}  # participant_id -> {signal_name -> SignalScore}
        
        for participant in participant_list:
            pid = participant.participant_id
            all_scores[pid] = {}
            
            for name, analyzer in self.analyzers.items():
                try:
                    score = analyzer.analyze(participant, self.context, participant_list)
                    all_scores[pid][name] = score
                except Exception as e:
                    logger.error(f"Error in {name} analyzer for {pid}: {e}")
                    all_scores[pid][name] = SignalScore(
                        signal_name=name,
                        participant_id=pid,
                        score=0.5,
                        confidence=0.0,
                        explanation=f"Analyzer error: {str(e)}"
                    )
        
        # Step 2: Bayesian fusion
        posteriors = self._bayesian_fusion(participant_list, all_scores)
        
        # Step 3: Build assessments with explanations
        assessments = []
        for participant in participant_list:
            pid = participant.participant_id
            
            # Generate explanations sorted by signal impact
            explanations = self._generate_explanations(pid, all_scores[pid], posteriors[pid])
            
            assessment = CandidateAssessment(
                participant_id=pid,
                display_name=participant.display_name,
                probability=posteriors[pid],
                signal_scores=all_scores[pid],
                explanations=explanations,
            )
            assessments.append(assessment)
        
        # Sort by probability (highest first)
        assessments.sort(key=lambda a: a.probability, reverse=True)
        
        # Step 4: Determine identification status
        top = assessments[0] if assessments else None
        second = assessments[1] if len(assessments) > 1 else None
        
        if top and top.probability >= CONFIDENCE_IDENTIFIED:
            status = "identified"
            identified_id = top.participant_id
            top.is_identified = True
            top.identified_at = time.time()
            
            margin = top.probability - (second.probability if second else 0)
            explanation = (
                f"Identified '{top.display_name}' as the candidate with "
                f"{top.probability:.0%} confidence (margin: {margin:.0%}). "
                f"{top.explanations[0] if top.explanations else ''}"
            )
        elif top and top.probability >= CONFIDENCE_LIKELY:
            status = "likely"
            identified_id = top.participant_id
            explanation = (
                f"'{top.display_name}' is likely the candidate ({top.probability:.0%} confidence) "
                f"but more evidence needed. {top.explanations[0] if top.explanations else ''}"
            )
        elif top and top.probability >= CONFIDENCE_UNCERTAIN:
            status = "analyzing"
            identified_id = None
            explanation = (
                f"Still analyzing — '{top.display_name}' leads at {top.probability:.0%} "
                f"but confidence is below threshold"
            )
        else:
            status = "uncertain"
            identified_id = None
            explanation = "Unable to determine candidate with current information"
        
        result = IdentificationResult(
            timestamp=time.time(),
            assessments=assessments,
            identified_candidate_id=identified_id,
            overall_confidence=top.probability if top else 0.0,
            status=status,
            explanation=explanation,
            event_log=self.event_history[-10:]  # last 10 events
        )
        
        self._current_result = result
        self.result_history.append(result)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return result
    
    def _bayesian_fusion(self, participants: list, all_scores: dict) -> dict:
        """
        Combine signal scores using a weighted log-odds Bayesian fusion scheme.
        
        This treats each signal's score as an evidence indicator:
        - score > 0.5: evidence supporting the participant being the candidate
        - score < 0.5: evidence opposing
        - score = 0.5: neutral
        
        By using the log-odds space, independent signals compound correctly, 
        and the scaling factor allows the engine to reach high-confidence 
        decision thresholds (>=75%) when multiple signals align.
        """
        n = len(participants)
        if n == 0:
            return {}
        
        # Start with 0 log-odds for each participant (uniform prior)
        log_odds = {p.participant_id: 0.0 for p in participants}
        
        # Update log-odds with each signal
        for signal_name, weight in self.signal_weights.items():
            for p in participants:
                pid = p.participant_id
                if signal_name in all_scores.get(pid, {}):
                    signal = all_scores[pid][signal_name]
                    
                    # Clamp score to avoid infinity
                    score = max(0.02, min(0.98, signal.score))
                    
                    # Log-odds of this signal: log(p / (1-p))
                    # score > 0.5 is positive (supporting), score < 0.5 is negative (opposing)
                    odds_update = np.log(score) - np.log(1.0 - score)
                    
                    # Apply weight and confidence as evidence scaling
                    log_odds[pid] += odds_update * signal.confidence * weight * 3.5
        
        # Convert log-odds back to probabilities using softmax/normalization
        max_log = max(log_odds.values())
        exp_odds = {pid: np.exp(lo - max_log) for pid, lo in log_odds.items()}
        total = sum(exp_odds.values())
        
        if total > 0:
            posteriors = {pid: eo / total for pid, eo in exp_odds.items()}
        else:
            posteriors = {p.participant_id: 1.0 / n for p in participants}
        
        return posteriors
    
    def _generate_explanations(self, participant_id: str, 
                                scores: dict, posterior: float) -> list:
        """
        Generate human-readable explanations for why a participant 
        was or wasn't identified as the candidate.
        
        Sorts by signal impact (how much each signal moved the probability).
        """
        explanations = []
        
        # Calculate signal impacts
        signal_impacts = []
        for signal_name, signal in scores.items():
            weight = self.signal_weights.get(signal_name, 0.1)
            impact = (signal.score - 0.5) * signal.confidence * weight
            
            if signal.explanation and signal.confidence > 0.1:
                signal_impacts.append((abs(impact), impact, signal))
        
        # Sort by absolute impact (most impactful first)
        signal_impacts.sort(key=lambda x: x[0], reverse=True)
        
        for _, impact, signal in signal_impacts:
            if signal.explanation:
                direction = "supports" if impact > 0 else "against"
                explanations.append(
                    f"[{signal.signal_name}] {signal.explanation}"
                )
        
        if not explanations:
            explanations.append("No significant signals detected yet")
        
        return explanations
    
    def get_signal_breakdown(self, participant_id: str) -> dict:
        """Get a detailed breakdown of all signals for a participant."""
        if not self._current_result:
            return {}
        
        for assessment in self._current_result.assessments:
            if assessment.participant_id == participant_id:
                breakdown = {}
                for signal_name, signal in assessment.signal_scores.items():
                    breakdown[signal_name] = {
                        "score": signal.score,
                        "confidence": signal.confidence,
                        "explanation": signal.explanation,
                        "weight": self.signal_weights.get(signal_name, 0),
                        "sub_scores": signal.sub_scores,
                    }
                return breakdown
        
        return {}
    
    def get_confidence_history(self) -> list:
        """Get the history of confidence scores over time."""
        history = []
        for result in self.result_history:
            entry = {
                "timestamp": result.timestamp,
                "status": result.status,
                "participants": {}
            }
            for assessment in result.assessments:
                entry["participants"][assessment.participant_id] = {
                    "display_name": assessment.display_name,
                    "probability": assessment.probability,
                }
            history.append(entry)
        return history
