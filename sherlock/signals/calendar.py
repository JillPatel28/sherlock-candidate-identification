"""
Calendar/Metadata Signal Analyzer

Cross-references participant information against external metadata:
- Calendar invite details
- Known interviewer lists
- Meeting role assignments
- Email domain analysis
"""

from sherlock.models import (
    Participant, MeetingContext, SignalScore
)
from rapidfuzz import fuzz


class CalendarAnalyzer:
    """
    Uses external metadata (calendar, email, schedule) to identify the candidate.
    
    Key signals:
    - Calendar invite roles (organizer vs attendee)
    - Email domain (company domain vs external)
    - Known interviewer list matching
    """
    
    SIGNAL_NAME = "calendar_metadata"
    
    def __init__(self):
        self.weight = 0.10
    
    def analyze(self, participant: Participant, context: MeetingContext,
                all_participants: list) -> SignalScore:
        """Analyze calendar/metadata signals for a participant."""
        
        sub_scores = {}
        explanations = []
        
        display_name = participant.display_name
        if participant.name_history:
            display_name = participant.name_history[-1]
        
        # --- Signal 1: Calendar invite role ---
        if context.calendar_invite_participants:
            # Check if this participant matches any calendar entry
            best_match = 0.0
            matched_role = None
            
            for cal_participant in context.calendar_invite_participants:
                cal_name = cal_participant.get("name", "")
                cal_email = cal_participant.get("email", "")
                cal_role = cal_participant.get("role", "attendee")
                
                # Match by name
                name_sim = fuzz.token_sort_ratio(display_name.lower(), cal_name.lower()) / 100.0
                
                if name_sim > best_match:
                    best_match = name_sim
                    matched_role = cal_role
            
            if best_match > 0.7 and matched_role:
                if matched_role == "organizer":
                    role_score = 0.2
                    explanations.append(f"Matched as meeting organizer in calendar — likely interviewer")
                elif matched_role == "candidate":
                    role_score = 0.9
                    explanations.append(f"Identified as candidate in calendar invite")
                elif matched_role == "interviewer":
                    role_score = 0.15
                    explanations.append(f"Listed as interviewer in calendar")
                elif matched_role == "observer":
                    role_score = 0.1
                    explanations.append(f"Listed as observer in calendar")
                else:
                    role_score = 0.5
                
                sub_scores["calendar_role"] = role_score
        
        # --- Signal 2: Known interviewer exclusion ---
        if context.interviewer_names:
            best_interviewer_match = 0.0
            matched_interviewer = None
            
            for int_name in context.interviewer_names:
                sim = fuzz.token_sort_ratio(display_name.lower(), int_name.lower()) / 100.0
                if sim > best_interviewer_match:
                    best_interviewer_match = sim
                    matched_interviewer = int_name
            
            if best_interviewer_match > 0.75:
                interviewer_score = 0.1
                explanations.append(f"Display name closely matches known interviewer '{matched_interviewer}'")
            elif best_interviewer_match > 0.5:
                interviewer_score = 0.3
            else:
                interviewer_score = 0.6  # not a known interviewer - slight positive signal
                explanations.append("Not matched to any known interviewer")
            
            sub_scores["interviewer_exclusion"] = interviewer_score
        
        # --- Signal 3: Email domain analysis ---
        if context.company_name:
            company_lower = context.company_name.lower()
            name_lower = display_name.lower()
            
            # If their name contains company name, likely an employee/interviewer
            if company_lower in name_lower:
                domain_score = 0.2
                explanations.append(f"Display name contains company name '{context.company_name}' — likely an employee")
            else:
                domain_score = 0.55
            
            sub_scores["company_match"] = domain_score
        
        # --- Signal 4: Participant count context ---
        active_count = sum(1 for p in all_participants if p.is_active)
        
        if active_count == 2:
            # 1-on-1 interview, participant is either candidate or interviewer
            count_score = 0.5  # neutral, need other signals
            explanations.append("1-on-1 meeting — equal probability without other signals")
        elif active_count == 1:
            count_score = 0.5
        elif active_count <= 4:
            # Small panel - typical interview setup
            count_score = 0.5
        else:
            # Large meeting - less typical interview format
            count_score = 0.45
        
        sub_scores["participant_context"] = count_score
        
        # Combine sub-signals
        weights = {
            "calendar_role": 0.35,
            "interviewer_exclusion": 0.30,
            "company_match": 0.20,
            "participant_context": 0.15,
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
        
        confidence = min(0.6, 0.1 + total_weight * 0.5)
        
        explanation = "; ".join(explanations) if explanations else "Limited calendar metadata available"
        
        return SignalScore(
            signal_name=self.SIGNAL_NAME,
            participant_id=participant.participant_id,
            score=final_score,
            confidence=confidence,
            explanation=explanation,
            sub_scores=sub_scores
        )
