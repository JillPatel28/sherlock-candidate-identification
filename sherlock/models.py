"""
Data models for the Sherlock candidate identification system.

Defines the core data structures used across all signal analyzers
and the fusion engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class EventType(Enum):
    """Types of meeting events we track."""
    JOIN = "join"
    LEAVE = "leave"
    WEBCAM_ON = "webcam_on"
    WEBCAM_OFF = "webcam_off"
    SCREEN_SHARE_START = "screen_share_start"
    SCREEN_SHARE_STOP = "screen_share_stop"
    SPEAKING_START = "speaking_start"
    SPEAKING_STOP = "speaking_stop"
    NAME_CHANGE = "name_change"
    CHAT_MESSAGE = "chat_message"


@dataclass
class MeetingEvent:
    """A single event that occurs during the meeting."""
    timestamp: float
    participant_id: str
    event_type: EventType
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Event({self.event_type.value}, participant={self.participant_id}, t={self.timestamp:.1f})"


@dataclass 
class SpeakingSegment:
    """A continuous segment where a participant is speaking."""
    participant_id: str
    start_time: float
    end_time: float
    transcript_text: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class TranscriptEntry:
    """A single transcript entry attributed to a speaker."""
    timestamp: float
    participant_id: str
    text: str
    confidence: float = 1.0


@dataclass
class Participant:
    """Represents a meeting participant with all known information."""
    participant_id: str
    display_name: str
    join_time: Optional[float] = None
    leave_time: Optional[float] = None
    is_active: bool = True
    webcam_on: bool = False
    is_screen_sharing: bool = False
    speaking_segments: list = field(default_factory=list)
    events: list = field(default_factory=list)
    name_history: list = field(default_factory=list)
    total_speaking_duration: float = 0.0
    speaking_turn_count: int = 0

    def __post_init__(self):
        if self.display_name and not self.name_history:
            self.name_history.append(self.display_name)

    def update_speaking_stats(self):
        """Recalculate speaking statistics from segments."""
        self.total_speaking_duration = sum(
            seg.duration for seg in self.speaking_segments
        )
        self.speaking_turn_count = len(self.speaking_segments)


@dataclass
class SignalScore:
    """
    Score from a single signal analyzer.
    
    Each analyzer produces a likelihood ratio for each participant:
    - score > 0.5 means evidence supports this participant being the candidate
    - score < 0.5 means evidence against
    - score = 0.5 means no information (neutral)
    
    The confidence field indicates how reliable this signal is (0-1).
    """
    signal_name: str
    participant_id: str
    score: float          # 0.0 to 1.0, probability this is the candidate
    confidence: float     # 0.0 to 1.0, how reliable this signal is
    explanation: str = ""
    sub_scores: dict = field(default_factory=dict)  # breakdown of components

    def __repr__(self):
        return f"Signal({self.signal_name}: {self.participant_id} = {self.score:.3f}, conf={self.confidence:.3f})"


@dataclass
class CandidateAssessment:
    """
    The fused assessment for a single participant.
    Combines all signal scores into a final probability.
    """
    participant_id: str
    display_name: str
    probability: float         # final posterior probability (0-1)
    signal_scores: dict = field(default_factory=dict)  # signal_name -> SignalScore
    explanations: list = field(default_factory=list)
    is_identified: bool = False
    identified_at: Optional[float] = None

    def top_explanation(self) -> str:
        """Get the most impactful explanation."""
        if self.explanations:
            return self.explanations[0]
        return "No evidence collected yet"


@dataclass
class MeetingContext:
    """
    All the external metadata we know about the meeting before it starts.
    This is the 'prior knowledge' that informs our analysis.
    """
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    interviewer_names: list = field(default_factory=list)
    scheduled_start_time: Optional[float] = None
    scheduled_end_time: Optional[float] = None
    calendar_invite_participants: list = field(default_factory=list)
    meeting_title: Optional[str] = None
    company_name: Optional[str] = None

    def get_candidate_name_parts(self) -> list:
        """Extract individual name parts for fuzzy matching."""
        parts = []
        if self.candidate_name:
            parts = self.candidate_name.lower().strip().split()
        return parts

    def get_email_name_parts(self) -> list:
        """Extract name parts from email address."""
        if not self.candidate_email:
            return []
        local_part = self.candidate_email.split("@")[0]
        # Handle common email formats: first.last, firstlast, first_last
        parts = local_part.replace(".", " ").replace("_", " ").replace("-", " ").split()
        return [p.lower() for p in parts]


@dataclass
class IdentificationResult:
    """The final output of the identification engine at any point in time."""
    timestamp: float
    assessments: list = field(default_factory=list)  # list of CandidateAssessment
    identified_candidate_id: Optional[str] = None
    overall_confidence: float = 0.0
    status: str = "analyzing"  # analyzing, identified, uncertain, no_candidate
    explanation: str = ""
    event_log: list = field(default_factory=list)

    def get_top_candidate(self) -> Optional[CandidateAssessment]:
        """Return the assessment with highest probability."""
        if not self.assessments:
            return None
        return max(self.assessments, key=lambda a: a.probability)
