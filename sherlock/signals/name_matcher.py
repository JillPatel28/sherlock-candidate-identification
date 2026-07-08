"""
Name Matching Signal Analyzer

Compares participant display names against known candidate information
using multiple fuzzy matching techniques. Handles common edge cases:
- Device names (e.g., "MacBook Pro", "iPhone")
- Nicknames and abbreviations
- Wrong/swapped names from interviewers
- Email-derived names
"""

from rapidfuzz import fuzz, process
from sherlock.models import (
    Participant, MeetingContext, SignalScore
)

# Common device/generic names that are definitely not a person
DEVICE_PATTERNS = [
    "macbook", "iphone", "ipad", "android", "pixel", "samsung",
    "laptop", "desktop", "phone", "tablet", "chromebook", "surface",
    "guest", "user", "unknown", "participant", "meeting room",
    "conference", "admin", "host", "organizer", "recorder",
]

# Common nickname mappings for fuzzy matching
NICKNAME_MAP = {
    "william": ["will", "bill", "billy", "liam"],
    "robert": ["rob", "bob", "bobby", "bert"],
    "richard": ["rick", "dick", "rich"],
    "james": ["jim", "jimmy", "jamie"],
    "john": ["jack", "johnny", "jon"],
    "elizabeth": ["liz", "beth", "lizzy", "eliza"],
    "jennifer": ["jen", "jenny"],
    "michael": ["mike", "mikey", "mick"],
    "katherine": ["kate", "kathy", "kat", "katie"],
    "christopher": ["chris", "topher"],
    "alexander": ["alex", "xander"],
    "nicholas": ["nick", "nico"],
    "benjamin": ["ben", "benny"],
    "daniel": ["dan", "danny"],
    "matthew": ["matt", "matty"],
    "joseph": ["joe", "joey"],
    "andrew": ["andy", "drew"],
    "joshua": ["josh"],
    "anthony": ["tony", "ant"],
    "thomas": ["tom", "tommy"],
    "david": ["dave", "davey"],
    "jonathan": ["jon", "jonny"],
    "samuel": ["sam", "sammy"],
    "patrick": ["pat", "paddy"],
    "margaret": ["maggie", "meg", "peggy"],
    "patricia": ["pat", "patty", "trish"],
    "rebecca": ["becca", "becky"],
    "stephanie": ["steph"],
    "victoria": ["vicky", "tori"],
    "priya": ["priya"],
    "jillian": ["jill"],
    "subramaniam": ["subra", "mani"],
}


def _is_device_name(name: str) -> bool:
    """Check if a display name looks like a device rather than a person."""
    name_lower = name.lower().strip()
    for pattern in DEVICE_PATTERNS:
        if pattern in name_lower:
            return True
    # Check if it's just numbers or very short non-name strings
    cleaned = name_lower.replace(" ", "").replace("-", "").replace("_", "")
    if cleaned.isdigit():
        return True
    if len(cleaned) <= 1:
        return True
    return False


def _check_nickname_match(name_part: str, candidate_part: str) -> float:
    """Check if two name parts could be nickname variants."""
    name_part = name_part.lower()
    candidate_part = candidate_part.lower()
    
    if name_part == candidate_part:
        return 1.0
    
    # Check both directions in nickname map
    for full_name, nicknames in NICKNAME_MAP.items():
        all_variants = [full_name] + nicknames
        if name_part in all_variants and candidate_part in all_variants:
            return 0.85
    
    return 0.0


def _compute_name_similarity(display_name: str, target_name: str) -> dict:
    """
    Compute multiple similarity metrics between a display name and target name.
    Returns a dict with individual scores and a combined score.
    """
    if not display_name or not target_name:
        return {"combined": 0.0, "exact": 0.0, "fuzzy": 0.0, "partial": 0.0, "token": 0.0}
    
    dn = display_name.lower().strip()
    tn = target_name.lower().strip()
    
    # Exact match
    exact = 1.0 if dn == tn else 0.0
    
    # Fuzzy ratio (overall similarity)
    fuzzy = fuzz.ratio(dn, tn) / 100.0
    
    # Partial ratio (handles substrings - good for "John" matching "John Smith")
    partial = fuzz.partial_ratio(dn, tn) / 100.0
    
    # Token sort ratio (handles reordered names - "Smith John" vs "John Smith")
    token_sort = fuzz.token_sort_ratio(dn, tn) / 100.0
    
    # Token set ratio (handles extra words - "Dr. John Smith PhD" vs "John Smith")
    token_set = fuzz.token_set_ratio(dn, tn) / 100.0
    
    # Check for nickname matches
    dn_parts = dn.split()
    tn_parts = tn.split()
    nickname_score = 0.0
    
    for dp in dn_parts:
        for tp in tn_parts:
            ns = _check_nickname_match(dp, tp)
            if ns > nickname_score:
                nickname_score = ns
    
    # Combined score - weighted combination of all metrics
    combined = max(
        exact,
        0.35 * fuzzy + 0.25 * partial + 0.20 * token_sort + 0.15 * token_set + 0.05 * nickname_score,
        nickname_score * 0.7 + partial * 0.3,
    )
    
    return {
        "combined": min(combined, 1.0),
        "exact": exact,
        "fuzzy": fuzzy,
        "partial": partial,
        "token_sort": token_sort,
        "token_set": token_set,
        "nickname": nickname_score,
    }


class NameMatchAnalyzer:
    """
    Analyzes participant display names against known candidate information.
    
    Produces a score indicating how likely each participant is the candidate
    based purely on their display name matching.
    """
    
    SIGNAL_NAME = "name_match"
    
    def __init__(self):
        self.weight = 0.30  # name matching is a strong but not definitive signal
    
    def analyze(self, participant: Participant, context: MeetingContext, 
                all_participants: list) -> SignalScore:
        """
        Analyze a single participant's name against candidate info.
        """
        display_name = participant.display_name
        
        # Use the most recent name if they changed it
        if participant.name_history:
            display_name = participant.name_history[-1]
        
        # Check if this is clearly a device name
        if _is_device_name(display_name):
            return SignalScore(
                signal_name=self.SIGNAL_NAME,
                participant_id=participant.participant_id,
                score=0.3,  # slightly below neutral - could still be candidate on device
                confidence=0.4,
                explanation=f"'{display_name}' appears to be a device name, not a person's name. Could still be the candidate using a device.",
                sub_scores={"is_device": 1.0}
            )
        
        scores = {}
        explanations = []
        
        # Compare against candidate name
        if context.candidate_name:
            name_sim = _compute_name_similarity(display_name, context.candidate_name)
            scores["candidate_name"] = name_sim
            
            if name_sim["exact"] == 1.0:
                explanations.append(f"Display name '{display_name}' exactly matches candidate name '{context.candidate_name}'")
            elif name_sim["combined"] > 0.7:
                explanations.append(f"Display name '{display_name}' closely matches candidate name '{context.candidate_name}' (similarity: {name_sim['combined']:.0%})")
            elif name_sim["nickname"] > 0:
                explanations.append(f"Display name '{display_name}' may be a nickname variant of '{context.candidate_name}'")
        
        # Compare against candidate email
        if context.candidate_email:
            email_parts = context.get_email_name_parts()
            email_name = " ".join(email_parts)
            email_sim = _compute_name_similarity(display_name, email_name)
            scores["email_name"] = email_sim
            
            if email_sim["combined"] > 0.6:
                explanations.append(f"Display name matches parts of candidate email '{context.candidate_email}'")
        
        # Check against interviewer names (negative signal)
        interviewer_match = 0.0
        for interviewer_name in context.interviewer_names:
            int_sim = _compute_name_similarity(display_name, interviewer_name)
            if int_sim["combined"] > interviewer_match:
                interviewer_match = int_sim["combined"]
                if int_sim["combined"] > 0.7:
                    explanations.append(f"Display name '{display_name}' matches known interviewer '{interviewer_name}'")
        
        # Compute final score
        candidate_score = 0.0
        if "candidate_name" in scores:
            candidate_score = max(candidate_score, scores["candidate_name"]["combined"])
        if "email_name" in scores:
            candidate_score = max(candidate_score, scores["email_name"]["combined"] * 0.8)
        
        # If they match an interviewer strongly, reduce score
        if interviewer_match > 0.7:
            final_score = max(0.05, candidate_score * 0.2)
            confidence = 0.8
            if not explanations:
                explanations.append(f"'{display_name}' matches a known interviewer name")
        elif candidate_score > 0.7:
            final_score = min(0.95, 0.5 + candidate_score * 0.45)
            confidence = min(0.9, candidate_score)
        elif candidate_score > 0.4:
            final_score = 0.4 + candidate_score * 0.3
            confidence = 0.5
        else:
            # No strong match in either direction
            final_score = 0.5  # neutral
            confidence = 0.3
            if not explanations:
                explanations.append(f"Display name '{display_name}' doesn't clearly match any known participant")
        
        explanation = "; ".join(explanations) if explanations else f"No strong name match for '{display_name}'"
        
        return SignalScore(
            signal_name=self.SIGNAL_NAME,
            participant_id=participant.participant_id,
            score=final_score,
            confidence=confidence,
            explanation=explanation,
            sub_scores={
                "candidate_name_sim": scores.get("candidate_name", {}).get("combined", 0),
                "email_sim": scores.get("email_name", {}).get("combined", 0),
                "interviewer_match": interviewer_match,
            }
        )
