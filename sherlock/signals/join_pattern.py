"""
Join Pattern Signal Analyzer

Analyzes when and how participants join the meeting to infer
who is likely the candidate. Key heuristics:
- Candidates typically join slightly before or right at start time
- Interviewers often join first to prepare
- Observers tend to join silently and late
- Candidates rarely leave and rejoin
"""

from sherlock.models import (
    Participant, MeetingContext, SignalScore, EventType
)


class JoinPatternAnalyzer:
    """
    Analyzes participant join/leave patterns to identify the candidate.
    
    Candidates exhibit specific join behaviors:
    1. They join around the scheduled time (not too early, not too late)
    2. They typically stay for the full interview
    3. They usually join exactly once
    4. They join after at least one interviewer is present
    """
    
    SIGNAL_NAME = "join_pattern"
    
    def __init__(self):
        self.weight = 0.15
    
    def analyze(self, participant: Participant, context: MeetingContext,
                all_participants: list) -> SignalScore:
        """Analyze join patterns for a participant."""
        
        if participant.join_time is None:
            return SignalScore(
                signal_name=self.SIGNAL_NAME,
                participant_id=participant.participant_id,
                score=0.5,
                confidence=0.1,
                explanation="No join time recorded"
            )
        
        sub_scores = {}
        explanations = []
        
        # --- Signal 1: Join order ---
        # Sort all participants by join time
        joined_participants = sorted(
            [p for p in all_participants if p.join_time is not None],
            key=lambda p: p.join_time
        )
        
        if joined_participants:
            join_position = next(
                (i for i, p in enumerate(joined_participants) 
                 if p.participant_id == participant.participant_id),
                -1
            )
            total = len(joined_participants)
            
            if total <= 1:
                order_score = 0.5
            elif join_position == 0:
                # First to join - more likely interviewer
                order_score = 0.3
                explanations.append("Joined first — more typical of an interviewer setting up")
            elif join_position == 1 and total >= 3:
                # Second to join - could be candidate arriving on time
                order_score = 0.65
                explanations.append("Joined second — consistent with candidate arriving on time")
            elif join_position == total - 1 and total >= 4:
                # Last to join in a large meeting - might be observer
                order_score = 0.35
                explanations.append("Joined last in a multi-participant meeting — may be an observer")
            else:
                order_score = 0.5
            
            sub_scores["join_order"] = order_score
        
        # --- Signal 2: Timing relative to schedule ---
        if context.scheduled_start_time and participant.join_time:
            delta = participant.join_time - context.scheduled_start_time
            
            if -120 <= delta <= 60:
                # Joined within 2 min before to 1 min after - typical candidate
                timing_score = 0.7
                explanations.append(f"Joined close to scheduled time ({delta:+.0f}s) — typical candidate behavior")
            elif -600 <= delta < -120:
                # Joined 2-10 min early - more likely interviewer
                timing_score = 0.35
                explanations.append(f"Joined {abs(delta):.0f}s early — more typical of interviewer preparation")
            elif 60 < delta <= 300:
                # Joined 1-5 min late - could be late candidate
                timing_score = 0.55
                explanations.append(f"Joined {delta:.0f}s late — could be a late candidate")
            elif delta > 300:
                # Joined very late - likely observer
                timing_score = 0.25
                explanations.append(f"Joined {delta:.0f}s after start — likely an observer")
            else:
                timing_score = 0.4
                explanations.append(f"Joined very early ({abs(delta):.0f}s before start)")
            
            sub_scores["timing"] = timing_score
        
        # --- Signal 3: Join/leave stability ---
        join_events = [e for e in participant.events if e.event_type == EventType.JOIN]
        leave_events = [e for e in participant.events if e.event_type == EventType.LEAVE]
        
        rejoin_count = len(join_events)
        if rejoin_count == 1:
            stability_score = 0.6
            explanations.append("Single stable session — typical of a candidate")
        elif rejoin_count == 2:
            stability_score = 0.5
            explanations.append("Rejoined once — possible connection issue")
        else:
            stability_score = 0.35
            explanations.append(f"Joined {rejoin_count} times — unusual for a candidate")
        
        sub_scores["stability"] = stability_score
        
        # --- Signal 4: Still in meeting (active) ---
        if participant.is_active:
            active_score = 0.6
        elif participant.leave_time:
            active_score = 0.35
            explanations.append("Has already left the meeting")
        else:
            active_score = 0.5
        
        sub_scores["active"] = active_score
        
        # Combine sub-signals
        weights = {"join_order": 0.25, "timing": 0.30, "stability": 0.25, "active": 0.20}
        
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
        
        confidence = min(0.6, total_weight * 0.7)
        
        explanation = "; ".join(explanations) if explanations else "No distinctive join pattern"
        
        return SignalScore(
            signal_name=self.SIGNAL_NAME,
            participant_id=participant.participant_id,
            score=final_score,
            confidence=confidence,
            explanation=explanation,
            sub_scores=sub_scores
        )
