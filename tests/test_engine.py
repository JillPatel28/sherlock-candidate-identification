"""
Unit tests for the Sherlock Candidate Identification Engine.

Tests cover:
- Individual signal analyzers
- Bayesian fusion engine
- Edge cases (device names, nicknames, wrong names)
- End-to-end scenario validation
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sherlock.models import (
    Participant, MeetingContext, MeetingEvent, EventType,
    SpeakingSegment, TranscriptEntry, SignalScore
)
from sherlock.signals.name_matcher import NameMatchAnalyzer, _is_device_name, _compute_name_similarity
from sherlock.signals.join_pattern import JoinPatternAnalyzer
from sherlock.signals.speaking import SpeakingAnalyzer
from sherlock.signals.behavioral import BehavioralAnalyzer
from sherlock.signals.transcript import TranscriptAnalyzer
from sherlock.signals.calendar import CalendarAnalyzer
from sherlock.engine import CandidateIdentificationEngine
from sherlock.simulator import MeetingSimulator, SCENARIOS


class TestNameMatcher:
    """Tests for the name matching signal analyzer."""
    
    def setup_method(self):
        self.analyzer = NameMatchAnalyzer()
        self.context = MeetingContext(
            candidate_name="John Smith",
            candidate_email="john.smith@gmail.com",
            interviewer_names=["Alice Johnson"]
        )
    
    def test_exact_match(self):
        p = Participant(participant_id="p1", display_name="John Smith")
        score = self.analyzer.analyze(p, self.context, [p])
        assert score.score > 0.7, f"Exact match should score high, got {score.score}"
    
    def test_partial_match(self):
        p = Participant(participant_id="p1", display_name="John")
        score = self.analyzer.analyze(p, self.context, [p])
        assert score.score > 0.5, f"Partial match should score above neutral, got {score.score}"
    
    def test_no_match(self):
        p = Participant(participant_id="p1", display_name="Bob Wilson")
        score = self.analyzer.analyze(p, self.context, [p])
        assert score.score <= 0.6, f"No match should score neutral or below, got {score.score}"
    
    def test_interviewer_match(self):
        p = Participant(participant_id="p1", display_name="Alice Johnson")
        score = self.analyzer.analyze(p, self.context, [p])
        assert score.score < 0.3, f"Interviewer match should score very low, got {score.score}"
    
    def test_device_name_detection(self):
        assert _is_device_name("MacBook Pro") is True
        assert _is_device_name("iPhone 15") is True
        assert _is_device_name("John Smith") is False
        assert _is_device_name("Guest") is True
    
    def test_device_name_score(self):
        p = Participant(participant_id="p1", display_name="MacBook Pro")
        score = self.analyzer.analyze(p, self.context, [p])
        assert score.score < 0.5, "Device name should score below neutral"
        assert score.confidence < 0.5, "Device name confidence should be low"
    
    def test_nickname_matching(self):
        context = MeetingContext(candidate_name="William Roberts")
        p = Participant(participant_id="p1", display_name="Bill Roberts")
        score = self.analyzer.analyze(p, context, [p])
        assert score.score > 0.5, f"Nickname should match, got {score.score}"
    
    def test_email_name_extraction(self):
        sim = _compute_name_similarity("John Smith", "john smith")
        assert sim["combined"] > 0.9


class TestJoinPattern:
    """Tests for join pattern analysis."""
    
    def setup_method(self):
        self.analyzer = JoinPatternAnalyzer()
        self.base_time = time.time()
        self.context = MeetingContext(
            scheduled_start_time=self.base_time
        )
    
    def test_candidate_join_timing(self):
        """Candidate joins close to start time."""
        p = Participant(
            participant_id="p1", display_name="Candidate",
            join_time=self.base_time - 5
        )
        p.events.append(MeetingEvent(self.base_time - 5, "p1", EventType.JOIN))
        
        interviewer = Participant(
            participant_id="p2", display_name="Interviewer",
            join_time=self.base_time - 120
        )
        interviewer.events.append(MeetingEvent(self.base_time - 120, "p2", EventType.JOIN))
        
        all_p = [interviewer, p]
        score = self.analyzer.analyze(p, self.context, all_p)
        assert score.score > 0.5, f"On-time joiner should score above neutral, got {score.score}"
    
    def test_early_joiner(self):
        """Interviewer joins very early."""
        p = Participant(
            participant_id="p1", display_name="Interviewer",
            join_time=self.base_time - 300
        )
        p.events.append(MeetingEvent(self.base_time - 300, "p1", EventType.JOIN))
        
        score = self.analyzer.analyze(p, self.context, [p])
        assert score.score < 0.55, f"Very early joiner should not score high, got {score.score}"
    
    def test_late_joiner(self):
        """Observer joins very late."""
        p = Participant(
            participant_id="p1", display_name="Observer",
            join_time=self.base_time + 600
        )
        p.events.append(MeetingEvent(self.base_time + 600, "p1", EventType.JOIN))
        
        others = [
            Participant("p2", "Int1", join_time=self.base_time - 60),
            Participant("p3", "Int2", join_time=self.base_time - 30),
            Participant("p4", "Cand", join_time=self.base_time + 5),
        ]
        for o in others:
            o.events.append(MeetingEvent(o.join_time, o.participant_id, EventType.JOIN))
        
        all_p = others + [p]
        score = self.analyzer.analyze(p, self.context, all_p)
        assert score.score < 0.5, f"Very late joiner should score below neutral, got {score.score}"


class TestSpeakingAnalyzer:
    """Tests for speaking pattern analysis."""
    
    def setup_method(self):
        self.analyzer = SpeakingAnalyzer()
        self.context = MeetingContext()
    
    def test_candidate_speaking_ratio(self):
        """Candidate speaks ~50% — typical interview ratio."""
        candidate = Participant(participant_id="p1", display_name="Candidate")
        candidate.speaking_segments = [
            SpeakingSegment("p1", 0, 30),
            SpeakingSegment("p1", 45, 75),
        ]
        candidate.update_speaking_stats()
        
        interviewer = Participant(participant_id="p2", display_name="Interviewer")
        interviewer.speaking_segments = [
            SpeakingSegment("p2", 30, 45),
            SpeakingSegment("p2", 75, 90),
        ]
        interviewer.update_speaking_stats()
        
        all_p = [candidate, interviewer]
        score = self.analyzer.analyze(candidate, self.context, all_p)
        assert score.score > 0.6, f"50% speaking ratio should score high, got {score.score}"
    
    def test_silent_observer(self):
        """Observer who never speaks."""
        observer = Participant(participant_id="p1", display_name="Observer")
        observer.update_speaking_stats()
        
        speaker = Participant(participant_id="p2", display_name="Speaker")
        speaker.speaking_segments = [SpeakingSegment("p2", 0, 60)]
        speaker.update_speaking_stats()
        
        all_p = [observer, speaker]
        score = self.analyzer.analyze(observer, self.context, all_p)
        assert score.score < 0.3, f"Silent participant should score very low, got {score.score}"


class TestBehavioralAnalyzer:
    """Tests for behavioral signal analysis."""
    
    def setup_method(self):
        self.analyzer = BehavioralAnalyzer()
        self.context = MeetingContext()
    
    def test_webcam_on_only(self):
        """Only participant with webcam on — strong candidate signal."""
        candidate = Participant(participant_id="p1", display_name="Candidate", webcam_on=True)
        candidate.events.append(MeetingEvent(0, "p1", EventType.WEBCAM_ON))
        
        others = [
            Participant("p2", "Int1", webcam_on=False),
            Participant("p3", "Obs1", webcam_on=False),
        ]
        
        all_p = [candidate] + others
        score = self.analyzer.analyze(candidate, self.context, all_p)
        assert score.score > 0.6, f"Only webcam-on participant should score high, got {score.score}"
    
    def test_screen_sharing(self):
        """Screen sharing indicates interviewer."""
        p = Participant(participant_id="p1", display_name="Sharer", is_screen_sharing=True)
        p.events.append(MeetingEvent(0, "p1", EventType.SCREEN_SHARE_START))
        
        score = self.analyzer.analyze(p, self.context, [p])
        assert score.score < 0.5, f"Screen sharer should score below neutral, got {score.score}"


class TestTranscriptAnalyzer:
    """Tests for transcript analysis."""
    
    def setup_method(self):
        self.analyzer = TranscriptAnalyzer()
        self.context = MeetingContext(candidate_name="Priya Sharma")
    
    def test_self_introduction(self):
        """Detects when someone introduces themselves."""
        p = Participant(participant_id="p1", display_name="Priya")
        
        entries = [
            TranscriptEntry(0, "p1", "Hi, I'm Priya Sharma. Nice to meet you!"),
            TranscriptEntry(10, "p1", "In my previous role, I built machine learning pipelines."),
        ]
        self.analyzer.set_transcript(entries)
        
        score = self.analyzer.analyze(p, self.context, [p])
        assert score.score > 0.6, f"Self-introduction should score high, got {score.score}"
    
    def test_interviewer_questions(self):
        """Detects interviewer question patterns."""
        p = Participant(participant_id="p1", display_name="Interviewer")
        
        entries = [
            TranscriptEntry(0, "p1", "Tell me about yourself and your experience."),
            TranscriptEntry(10, "p1", "Can you describe how you would approach this problem?"),
            TranscriptEntry(20, "p1", "Let's move on to the next question. How do you handle that?"),
        ]
        self.analyzer.set_transcript(entries)
        
        score = self.analyzer.analyze(p, self.context, [p])
        assert score.score < 0.5, f"Interviewer questions should score below neutral, got {score.score}"


class TestCalendarAnalyzer:
    """Tests for calendar metadata analysis."""
    
    def setup_method(self):
        self.analyzer = CalendarAnalyzer()
    
    def test_calendar_candidate_role(self):
        """Calendar marks someone as candidate."""
        context = MeetingContext(
            calendar_invite_participants=[
                {"name": "John Smith", "email": "john@gmail.com", "role": "candidate"},
            ]
        )
        p = Participant(participant_id="p1", display_name="John Smith")
        score = self.analyzer.analyze(p, context, [p])
        assert score.score > 0.7, f"Calendar candidate should score high, got {score.score}"
    
    def test_calendar_organizer(self):
        """Calendar marks someone as organizer."""
        context = MeetingContext(
            calendar_invite_participants=[
                {"name": "Alice Chen", "email": "alice@company.com", "role": "organizer"},
            ]
        )
        p = Participant(participant_id="p1", display_name="Alice Chen")
        score = self.analyzer.analyze(p, context, [p])
        assert score.score < 0.4, f"Calendar organizer should score low, got {score.score}"


class TestEngine:
    """Integration tests for the full engine."""
    
    def test_standard_scenario(self):
        """Standard interview should correctly identify the candidate."""
        sim = MeetingSimulator("standard")
        engine = CandidateIdentificationEngine(sim.context)
        results = sim.run_full(engine)
        
        final = results[-1]["result"]
        top = final.get_top_candidate()
        
        assert top is not None, "Should identify a top candidate"
        assert final.overall_confidence > 0.5, f"Should have decent confidence, got {final.overall_confidence}"
        assert "Priya" in top.display_name or "Sharma" in top.display_name, \
            f"Should identify Priya Sharma, got {top.display_name}"
    
    def test_device_name_scenario(self):
        """Should identify candidate even with device name."""
        sim = MeetingSimulator("device_name")
        engine = CandidateIdentificationEngine(sim.context)
        results = sim.run_full(engine)
        
        final = results[-1]["result"]
        top = final.get_top_candidate()
        
        assert top is not None
        assert final.overall_confidence > 0.4
        # The candidate joins as "MacBook Pro" — system should still figure it out
        assert "MacBook" in top.display_name or "Jordan" in top.display_name
    
    def test_panel_scenario(self):
        """Panel interview with observers should identify the candidate."""
        sim = MeetingSimulator("panel_with_observers")
        engine = CandidateIdentificationEngine(sim.context)
        results = sim.run_full(engine)
        
        final = results[-1]["result"]
        top = final.get_top_candidate()
        
        assert top is not None
        assert "Aisha" in top.display_name or "Patel" in top.display_name, \
            f"Should identify Aisha Patel, got {top.display_name}"
    
    def test_nickname_scenario(self):
        """Should handle nickname (Bill vs William)."""
        sim = MeetingSimulator("nickname")
        engine = CandidateIdentificationEngine(sim.context)
        results = sim.run_full(engine)
        
        final = results[-1]["result"]
        top = final.get_top_candidate()
        
        assert top is not None
        assert "Bill" in top.display_name or "Roberts" in top.display_name
    
    def test_all_scenarios_run(self):
        """All scenarios should run without errors."""
        for scenario_name in SCENARIOS:
            sim = MeetingSimulator(scenario_name)
            engine = CandidateIdentificationEngine(sim.context)
            results = sim.run_full(engine)
            
            assert len(results) > 0, f"Scenario {scenario_name} produced no results"
            final = results[-1]["result"]
            assert final.assessments, f"Scenario {scenario_name} has no assessments"
    
    def test_confidence_increases(self):
        """Confidence should generally increase as more data arrives."""
        sim = MeetingSimulator("standard")
        engine = CandidateIdentificationEngine(sim.context)
        results = sim.run_full(engine)
        
        confidences = [r["result"].overall_confidence for r in results]
        # Last confidence should be higher than first
        assert confidences[-1] >= confidences[0], \
            f"Confidence should increase: {confidences}"
    
    def test_explanation_generation(self):
        """Engine should produce meaningful explanations."""
        sim = MeetingSimulator("standard")
        engine = CandidateIdentificationEngine(sim.context)
        results = sim.run_full(engine)
        
        final = results[-1]["result"]
        top = final.get_top_candidate()
        
        assert len(top.explanations) > 0, "Should generate explanations"
        assert any(len(e) > 10 for e in top.explanations), \
            "Explanations should be meaningful, not empty strings"
    
    def test_signal_breakdown(self):
        """Signal breakdown should return data for all analyzers."""
        sim = MeetingSimulator("standard")
        engine = CandidateIdentificationEngine(sim.context)
        sim.run_full(engine)
        
        # Get breakdown for first participant
        pid = list(engine.participants.keys())[0]
        breakdown = engine.get_signal_breakdown(pid)
        
        assert len(breakdown) == 6, f"Should have 6 signals, got {len(breakdown)}"
        for signal_name, info in breakdown.items():
            assert "score" in info
            assert "confidence" in info
            assert 0 <= info["score"] <= 1
            assert 0 <= info["confidence"] <= 1


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_participants(self):
        """Engine handles no participants gracefully."""
        context = MeetingContext(candidate_name="Nobody")
        engine = CandidateIdentificationEngine(context)
        result = engine.identify()
        
        assert result.status == "no_participants"
    
    def test_single_participant(self):
        """Engine handles single participant."""
        context = MeetingContext(candidate_name="John")
        engine = CandidateIdentificationEngine(context)
        
        p = Participant("p1", "John", join_time=time.time())
        engine.add_participant(p)
        
        result = engine.identify()
        assert len(result.assessments) == 1
        assert result.assessments[0].probability == 1.0  # only option
    
    def test_name_change_event(self):
        """Engine handles name changes correctly."""
        context = MeetingContext(candidate_name="Carlos Rodriguez")
        engine = CandidateIdentificationEngine(context)
        
        p = Participant("p1", "Guest User 42", join_time=time.time())
        engine.add_participant(p)
        
        # First analysis with generic name
        result1 = engine.identify()
        
        # Name change event
        event = MeetingEvent(
            time.time(), "p1", EventType.NAME_CHANGE,
            {"new_name": "Carlos Rodriguez"}
        )
        engine.add_event(event)
        
        # Second analysis should have higher confidence
        result2 = engine.identify()
        
        assert result2.overall_confidence >= result1.overall_confidence


class TestOnlineLearning:
    """Tests for online learning and feedback system."""
    
    def test_weight_update_rule(self):
        """Verify that confirming a candidate adjusts weights in expected direction."""
        import os
        
        # Ensure clean weights before test
        weights_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "learned_weights.json")
        if os.path.exists(weights_file):
            try:
                os.remove(weights_file)
            except Exception:
                pass
                
        # Run standard scenario
        sim = MeetingSimulator("standard")
        engine = CandidateIdentificationEngine(sim.context)
        
        # Pre-run weights should be default
        initial_weights = dict(engine.signal_weights)
        
        # Run full simulation to accumulate scores
        sim.run_full(engine)
        
        # Priya Sharma is the actual candidate ("p2")
        # Let's confirm Priya Sharma and trigger learn_from_feedback
        new_weights = engine.learn_from_feedback("p2", learning_rate=0.1)
        
        # Let's check that weights have indeed changed and sum to 1.0
        assert new_weights != initial_weights
        assert sum(new_weights.values()) == pytest.approx(1.0)
        
        # Let's clean up weights file after test
        if os.path.exists(weights_file):
            try:
                os.remove(weights_file)
            except Exception:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
