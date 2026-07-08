"""
Speaking Pattern Signal Analyzer

Analyzes how participants speak during the meeting to identify
the candidate. Key patterns:
- In interviews, candidates speak MORE than any single interviewer
- Candidates respond to questions (reactive speaking pattern)
- Candidates have longer speaking turns (answering questions in depth)
- Observers typically don't speak at all
"""

from sherlock.models import (
    Participant, MeetingContext, SignalScore
)


class SpeakingAnalyzer:
    """
    Analyzes speaking patterns to identify the candidate.
    
    In a typical interview:
    - Candidate speaks 40-60% of the time (answering questions)
    - Main interviewer speaks 30-50% (asking questions)
    - Additional interviewers speak 5-15% each
    - Observers speak 0-2%
    """
    
    SIGNAL_NAME = "speaking_pattern"
    
    def __init__(self):
        self.weight = 0.20
    
    def analyze(self, participant: Participant, context: MeetingContext,
                all_participants: list) -> SignalScore:
        """Analyze speaking patterns for a participant."""
        
        # Calculate total speaking time across all participants
        total_speaking_all = sum(
            p.total_speaking_duration for p in all_participants
        )
        
        sub_scores = {}
        explanations = []
        
        if total_speaking_all == 0:
            return SignalScore(
                signal_name=self.SIGNAL_NAME,
                participant_id=participant.participant_id,
                score=0.5,
                confidence=0.1,
                explanation="No speaking data available yet",
                sub_scores={"speaking_ratio": 0}
            )
        
        # --- Signal 1: Speaking ratio ---
        speaking_ratio = participant.total_speaking_duration / total_speaking_all
        
        # Candidates typically speak 35-65% in an interview
        if 0.35 <= speaking_ratio <= 0.65:
            ratio_score = 0.8
            explanations.append(f"Speaking {speaking_ratio:.0%} of total — consistent with candidate role")
        elif 0.25 <= speaking_ratio < 0.35:
            ratio_score = 0.6
            explanations.append(f"Speaking {speaking_ratio:.0%} — moderate, could be candidate or active interviewer")
        elif 0.65 < speaking_ratio <= 0.80:
            ratio_score = 0.65
            explanations.append(f"Speaking {speaking_ratio:.0%} — high ratio, likely candidate in a 1-on-1")
        elif speaking_ratio > 0.80:
            ratio_score = 0.45
            explanations.append(f"Speaking {speaking_ratio:.0%} — unusually dominant, might be presenting or an interviewer")
        elif 0.10 <= speaking_ratio < 0.25:
            ratio_score = 0.4
            explanations.append(f"Speaking only {speaking_ratio:.0%} — relatively quiet, less likely candidate")
        elif speaking_ratio < 0.05:
            ratio_score = 0.15
            explanations.append(f"Speaking {speaking_ratio:.0%} — nearly silent, likely an observer")
        else:
            ratio_score = 0.35
        
        sub_scores["speaking_ratio"] = ratio_score
        sub_scores["raw_ratio"] = speaking_ratio
        
        # --- Signal 2: Average turn duration ---
        if participant.speaking_turn_count > 0:
            avg_turn = participant.total_speaking_duration / participant.speaking_turn_count
            
            # Candidates tend to have longer turns (answering questions at length)
            if avg_turn > 20:
                turn_score = 0.75
                explanations.append(f"Average speaking turn of {avg_turn:.0f}s — long answers typical of candidate")
            elif avg_turn > 10:
                turn_score = 0.65
                explanations.append(f"Average speaking turn of {avg_turn:.0f}s — moderate length responses")
            elif avg_turn > 5:
                turn_score = 0.5
                explanations.append(f"Average speaking turn of {avg_turn:.0f}s — short responses")
            else:
                turn_score = 0.35
                explanations.append(f"Average speaking turn of {avg_turn:.0f}s — very brief, typical of interviewer questions")
            
            sub_scores["turn_duration"] = turn_score
        
        # --- Signal 3: Number of speaking turns ---
        if participant.speaking_turn_count > 0:
            # Compare against others
            max_turns = max(p.speaking_turn_count for p in all_participants)
            
            if max_turns > 0:
                relative_turns = participant.speaking_turn_count / max_turns
                
                if relative_turns > 0.7:
                    turns_score = 0.65  # active participant
                elif relative_turns > 0.3:
                    turns_score = 0.55
                else:
                    turns_score = 0.35
                
                sub_scores["turn_count"] = turns_score
        
        # --- Signal 4: Speaking recency (are they still talking?) ---
        if participant.speaking_segments:
            latest_segment = max(participant.speaking_segments, key=lambda s: s.end_time)
            # Check if they spoke recently (within last 60 seconds of available data)
            all_end_times = []
            for p in all_participants:
                if p.speaking_segments:
                    all_end_times.append(max(s.end_time for s in p.speaking_segments))
            
            if all_end_times:
                latest_overall = max(all_end_times)
                recency = latest_overall - latest_segment.end_time
                
                if recency < 30:
                    recency_score = 0.6
                elif recency < 120:
                    recency_score = 0.5
                else:
                    recency_score = 0.35
                
                sub_scores["recency"] = recency_score
        
        # Combine sub-signals
        weights = {
            "speaking_ratio": 0.40,
            "turn_duration": 0.25,
            "turn_count": 0.15,
            "recency": 0.20,
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
        
        # Confidence increases with more speaking data
        data_richness = min(1.0, participant.total_speaking_duration / 60.0)  # caps at 1 min
        confidence = min(0.75, 0.2 + data_richness * 0.55)
        
        explanation = "; ".join(explanations) if explanations else "Insufficient speaking data"
        
        return SignalScore(
            signal_name=self.SIGNAL_NAME,
            participant_id=participant.participant_id,
            score=final_score,
            confidence=confidence,
            explanation=explanation,
            sub_scores=sub_scores
        )
