"""
Transcript Signal Analyzer

Analyzes the meeting transcript to identify the candidate through
linguistic patterns. Key signals:
- Self-introductions ("I'm..." / "My name is...")
- Answering behavioral questions ("In my experience..." / "At my previous company...")
- Interview-specific language patterns
- Question vs answer ratio
"""

import re
from sherlock.models import (
    Participant, MeetingContext, SignalScore, TranscriptEntry
)


# Patterns that suggest someone is introducing themselves as candidate
SELF_INTRO_PATTERNS = [
    r"\b(?:i'?m|my name is|this is)\s+(\w+)",
    r"\b(?:hi|hello|hey),?\s*(?:i'?m|my name is)\s+(\w+)",
    r"\b(?:nice to meet|pleasure to meet|good to (?:meet|see))\b",
]

# Patterns that suggest someone is answering interview questions
CANDIDATE_ANSWER_PATTERNS = [
    r"\b(?:in my (?:previous|last|current) (?:role|job|position|company))\b",
    r"\b(?:i have (?:experience|worked|been|built|developed))\b",
    r"\b(?:my (?:background|experience|expertise) (?:is|includes))\b",
    r"\b(?:at (?:my (?:last|previous|current)|a previous) (?:company|role|job))\b",
    r"\b(?:i (?:led|managed|built|designed|implemented|created|developed|architected))\b",
    r"\b(?:i (?:graduated|studied|majored) (?:from|in|at))\b",
    r"\b(?:one (?:example|time|project) (?:where|when|that))\b",
    r"\b(?:the (?:challenge|situation|task|result|outcome) was)\b",
    r"\b(?:i would (?:approach|handle|solve|tackle|design|build))\b",
    r"\b(?:my (?:strengths?|weaknesses?|approach) (?:is|are|would be))\b",
    r"\b(?:i'?m (?:passionate|interested|excited) about)\b",
    r"\b(?:my resume|my portfolio|my github|my linkedin)\b",
]

# Patterns that suggest someone is asking interview questions
INTERVIEWER_QUESTION_PATTERNS = [
    r"\b(?:tell me about (?:yourself|a time|your))\b",
    r"\b(?:can you (?:describe|explain|walk me|tell me))\b",
    r"\b(?:what (?:is|are|was|were) your (?:experience|role|responsibility))\b",
    r"\b(?:how (?:would|did|do) you (?:handle|approach|solve|deal))\b",
    r"\b(?:why (?:are you|did you|do you) (?:interested|leave|want|looking))\b",
    r"\b(?:let me (?:explain|describe|tell you about) (?:the|our|this))\b",
    r"\b(?:(?:at|here at) (?:our company|the company|sherlock))\b",
    r"\b(?:the (?:role|position|job|team) (?:is|involves|requires))\b",
    r"\b(?:do you have any questions)\b",
    r"\b(?:let's (?:move on|start|begin|discuss|talk about))\b",
]


class TranscriptAnalyzer:
    """
    Analyzes meeting transcript to identify candidate through language patterns.
    
    Candidates use distinctly different language than interviewers:
    - They introduce themselves
    - They answer questions using past experience
    - They describe their skills and background
    - They use more self-referential language
    """
    
    SIGNAL_NAME = "transcript"
    
    def __init__(self):
        self.weight = 0.15
        self._transcript_entries = []
    
    def set_transcript(self, entries: list):
        """Update the transcript entries."""
        self._transcript_entries = entries
    
    def analyze(self, participant: Participant, context: MeetingContext,
                all_participants: list) -> SignalScore:
        """Analyze transcript content for a participant."""
        
        # Get this participant's transcript entries
        entries = [
            e for e in self._transcript_entries 
            if e.participant_id == participant.participant_id
        ]
        
        if not entries:
            return SignalScore(
                signal_name=self.SIGNAL_NAME,
                participant_id=participant.participant_id,
                score=0.5,
                confidence=0.1,
                explanation="No transcript entries for this participant",
                sub_scores={"entry_count": 0}
            )
        
        sub_scores = {}
        explanations = []
        
        all_text = " ".join(e.text.lower() for e in entries)
        total_words = len(all_text.split())
        
        # --- Signal 1: Self-introduction detection ---
        intro_matches = 0
        for pattern in SELF_INTRO_PATTERNS:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            intro_matches += len(matches)
            
            # Check if the introduced name matches candidate name
            if matches and context.candidate_name:
                for match in matches:
                    if isinstance(match, str):
                        name = match.lower()
                        if name in context.candidate_name.lower():
                            intro_matches += 3  # strong boost
                            explanations.append(f"Introduced themselves with a name matching candidate: '{match}'")
        
        if intro_matches > 0:
            intro_score = min(0.85, 0.5 + intro_matches * 0.1)
            if not any("Introduced" in e for e in explanations):
                explanations.append("Used self-introduction language")
        else:
            intro_score = 0.5
        
        sub_scores["self_introduction"] = intro_score
        
        # --- Signal 2: Candidate answer patterns ---
        answer_matches = 0
        for pattern in CANDIDATE_ANSWER_PATTERNS:
            answer_matches += len(re.findall(pattern, all_text, re.IGNORECASE))
        
        if total_words > 0:
            answer_density = answer_matches / (total_words / 100)  # per 100 words
        else:
            answer_density = 0
        
        if answer_density > 2:
            answer_score = 0.85
            explanations.append(f"Frequently uses candidate-type language ({answer_matches} answer patterns)")
        elif answer_density > 1:
            answer_score = 0.7
            explanations.append(f"Uses some interview answer patterns ({answer_matches} matches)")
        elif answer_density > 0.3:
            answer_score = 0.55
        else:
            answer_score = 0.45
        
        sub_scores["answer_patterns"] = answer_score
        
        # --- Signal 3: Interviewer question patterns (negative signal) ---
        question_matches = 0
        for pattern in INTERVIEWER_QUESTION_PATTERNS:
            question_matches += len(re.findall(pattern, all_text, re.IGNORECASE))
        
        if total_words > 0:
            question_density = question_matches / (total_words / 100)
        else:
            question_density = 0
        
        if question_density > 2:
            question_score = 0.2
            explanations.append(f"Frequently asks interview-style questions ({question_matches} matches) — likely an interviewer")
        elif question_density > 1:
            question_score = 0.35
        else:
            question_score = 0.55
        
        sub_scores["question_patterns"] = question_score
        
        # --- Signal 4: Response length and depth ---
        avg_entry_length = total_words / len(entries)
        
        if avg_entry_length > 40:
            depth_score = 0.75
            explanations.append(f"Gives detailed responses (avg {avg_entry_length:.0f} words) — typical candidate behavior")
        elif avg_entry_length > 20:
            depth_score = 0.6
        elif avg_entry_length > 10:
            depth_score = 0.5
        else:
            depth_score = 0.35
            explanations.append(f"Short responses (avg {avg_entry_length:.0f} words) — more typical of interviewer questions")
        
        sub_scores["response_depth"] = depth_score
        
        # Combine sub-signals
        weights = {
            "self_introduction": 0.30,
            "answer_patterns": 0.30,
            "question_patterns": 0.20,
            "response_depth": 0.20,
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
        
        # Confidence increases with more transcript data
        data_richness = min(1.0, total_words / 200.0)
        confidence = min(0.7, 0.15 + data_richness * 0.55)
        
        explanation = "; ".join(explanations) if explanations else "Insufficient transcript data"
        
        return SignalScore(
            signal_name=self.SIGNAL_NAME,
            participant_id=participant.participant_id,
            score=final_score,
            confidence=confidence,
            explanation=explanation,
            sub_scores=sub_scores
        )
