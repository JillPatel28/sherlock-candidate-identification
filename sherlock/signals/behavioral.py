"""
Behavioral Signal Analyzer

Analyzes participant behaviors like webcam usage and screen sharing
to identify the candidate. Key patterns:
- Candidates almost always have their webcam ON
- Candidates rarely screen share (interviewers do for coding problems)
- Observers typically have webcam OFF
- Candidates stay focused and present throughout
"""

from sherlock.models import (
    Participant, MeetingContext, SignalScore, EventType
)


class BehavioralAnalyzer:
    """
    Analyzes webcam and screen sharing behaviors.
    
    In typical interviews:
    - Candidates keep webcam ON throughout (they want to make a good impression)
    - Interviewers may or may not have webcam on
    - Observers usually have webcam OFF
    - Interviewers screen share for coding exercises, not candidates
    """
    
    SIGNAL_NAME = "behavioral"
    
    def __init__(self):
        self.weight = 0.10
    
    def analyze(self, participant: Participant, context: MeetingContext,
                all_participants: list) -> SignalScore:
        """Analyze behavioral signals for a participant."""
        
        sub_scores = {}
        explanations = []
        
        # --- Signal 1: Webcam status ---
        webcam_events = [
            e for e in participant.events 
            if e.event_type in (EventType.WEBCAM_ON, EventType.WEBCAM_OFF)
        ]
        
        # Count how many participants have webcam on
        webcam_on_count = sum(1 for p in all_participants if p.webcam_on)
        total_active = sum(1 for p in all_participants if p.is_active)
        
        if participant.webcam_on:
            if webcam_on_count == 1:
                # Only person with webcam on - strong candidate signal
                webcam_score = 0.80
                explanations.append("Only participant with webcam on — strong candidate indicator")
            elif webcam_on_count <= total_active // 2:
                # Minority have webcam on
                webcam_score = 0.65
                explanations.append("Webcam is on — candidates typically keep webcam on")
            else:
                webcam_score = 0.55
                explanations.append("Webcam is on")
        else:
            if webcam_on_count > 0:
                webcam_score = 0.30
                explanations.append("Webcam is off — unusual for a candidate")
            else:
                # No one has webcam on, not informative
                webcam_score = 0.5
                explanations.append("Webcam is off, but no one has webcam on")
        
        sub_scores["webcam"] = webcam_score
        
        # --- Signal 2: Screen sharing ---
        screen_events = [
            e for e in participant.events
            if e.event_type in (EventType.SCREEN_SHARE_START, EventType.SCREEN_SHARE_STOP)
        ]
        
        if participant.is_screen_sharing:
            # Candidates rarely screen share (interviewers do for coding)
            screen_score = 0.35
            explanations.append("Currently screen sharing — more typical of an interviewer")
        elif screen_events:
            screen_score = 0.40
            explanations.append("Has screen shared — slightly more typical of interviewer")
        else:
            screen_score = 0.55
            explanations.append("Has not screen shared — consistent with candidate role")
        
        sub_scores["screen_share"] = screen_score
        
        # --- Signal 3: Webcam consistency ---
        if webcam_events:
            # How many times did they toggle webcam?
            toggle_count = len(webcam_events)
            if toggle_count <= 2:
                consistency_score = 0.6  # stable - turned on once, maybe adjusted
                explanations.append("Stable webcam usage throughout meeting")
            elif toggle_count <= 4:
                consistency_score = 0.5
            else:
                consistency_score = 0.35
                explanations.append("Frequent webcam toggling — less typical of candidate")
            
            sub_scores["webcam_consistency"] = consistency_score
        
        # --- Signal 4: Engagement score ---
        # Candidates are typically very engaged (webcam on + speaking + staying)
        engagement = 0.0
        engagement_factors = 0
        
        if participant.webcam_on:
            engagement += 1
        engagement_factors += 1
        
        if participant.total_speaking_duration > 0:
            engagement += 1
        engagement_factors += 1
        
        if participant.is_active:
            engagement += 1
        engagement_factors += 1
        
        if not participant.is_screen_sharing:
            engagement += 0.5
        engagement_factors += 0.5
        
        engagement_ratio = engagement / engagement_factors if engagement_factors > 0 else 0.5
        engagement_score = 0.3 + engagement_ratio * 0.4  # map to 0.3-0.7 range
        sub_scores["engagement"] = engagement_score
        
        # Combine sub-signals
        weights = {
            "webcam": 0.35,
            "screen_share": 0.25,
            "webcam_consistency": 0.15,
            "engagement": 0.25,
        }
        
        final_score = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            if key in sub_scores:
                final_score += sub_scores[key] * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score /= total_weight
        else:
            final_score = 0.5
        
        confidence = min(0.5, 0.2 + len(sub_scores) * 0.1)
        
        explanation = "; ".join(explanations) if explanations else "No behavioral signals available"
        
        return SignalScore(
            signal_name=self.SIGNAL_NAME,
            participant_id=participant.participant_id,
            score=final_score,
            confidence=confidence,
            explanation=explanation,
            sub_scores=sub_scores
        )
