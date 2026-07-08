"""
Meeting Simulator

Generates realistic meeting scenarios for demonstrating and testing the
candidate identification system. Each scenario simulates real-world
edge cases that Sherlock needs to handle.

Scenarios:
1. Standard interview - candidate name matches
2. Candidate uses device name (MacBook Pro)
3. Candidate uses nickname
4. Wrong name in system
5. Multiple observers + panel interview
6. Candidate changes display name mid-meeting
7. All ambiguous (stress test)
"""

import time
import random
from sherlock.models import (
    Participant, MeetingEvent, MeetingContext, EventType,
    SpeakingSegment, TranscriptEntry
)


def _make_event(timestamp, pid, event_type, metadata=None):
    return MeetingEvent(
        timestamp=timestamp,
        participant_id=pid,
        event_type=event_type,
        metadata=metadata or {}
    )


class MeetingSimulator:
    """
    Simulates meeting events over time for demo purposes.
    
    Each tick of the simulation generates new events, speaking segments,
    and transcript entries that the engine processes in real-time.
    """
    
    def __init__(self, scenario_name="standard"):
        self.scenario_name = scenario_name
        self.base_time = time.time()
        self.current_time = self.base_time
        self.participants = []
        self.events = []
        self.speaking_segments = []
        self.transcript_entries = []
        self.context = None
        self._setup_scenario(scenario_name)
    
    def _setup_scenario(self, name):
        """Initialize scenario data."""
        scenarios = {
            "standard": self._scenario_standard,
            "device_name": self._scenario_device_name,
            "nickname": self._scenario_nickname,
            "wrong_name": self._scenario_wrong_name,
            "panel_with_observers": self._scenario_panel_observers,
            "name_change": self._scenario_name_change,
            "ambiguous": self._scenario_ambiguous,
        }
        
        setup_fn = scenarios.get(name, self._scenario_standard)
        setup_fn()
    
    def _scenario_standard(self):
        """Standard interview: candidate name matches, 1-on-1."""
        self.context = MeetingContext(
            candidate_name="Priya Sharma",
            candidate_email="priya.sharma@gmail.com",
            interviewer_names=["Alex Chen"],
            scheduled_start_time=self.base_time,
            meeting_title="Technical Interview - Priya Sharma",
            company_name="Sherlock",
            calendar_invite_participants=[
                {"name": "Alex Chen", "email": "alex.chen@sherlock.ai", "role": "organizer"},
                {"name": "Priya Sharma", "email": "priya.sharma@gmail.com", "role": "candidate"},
            ]
        )
        
        self.timeline = [
            # Phase 1: Join events (t=0 to t=30s)
            {"time": -120, "action": "join", "pid": "p1", "name": "Alex Chen"},
            {"time": -120, "action": "webcam_on", "pid": "p1"},
            {"time": -5, "action": "join", "pid": "p2", "name": "Priya Sharma"},
            {"time": -5, "action": "webcam_on", "pid": "p2"},
            
            # Phase 2: Introductions (t=30s to t=90s)
            {"time": 5, "action": "speak", "pid": "p1", "duration": 8,
             "text": "Hi Priya, welcome! I'm Alex, I'll be conducting your technical interview today. How are you doing?"},
            {"time": 15, "action": "speak", "pid": "p2", "duration": 12,
             "text": "Hi Alex, nice to meet you! I'm doing great, thanks for asking. I'm really excited about this opportunity at Sherlock."},
            
            # Phase 3: Interview questions (t=90s to t=300s)
            {"time": 30, "action": "speak", "pid": "p1", "duration": 10,
             "text": "Great! Let's start. Can you tell me about your background and what drew you to this role?"},
            {"time": 45, "action": "speak", "pid": "p2", "duration": 45,
             "text": "Sure! My background is in machine learning and computer vision. In my previous role at TechCorp, I built real-time fraud detection systems processing millions of transactions. I led a team of four engineers and we reduced false positives by 40%. I'm passionate about applying AI to solve real-world security problems, which is exactly what Sherlock does."},
            
            {"time": 95, "action": "speak", "pid": "p1", "duration": 8,
             "text": "That's impressive. How would you approach building a system to detect deepfakes in real-time video calls?"},
            {"time": 108, "action": "speak", "pid": "p2", "duration": 50,
             "text": "Great question. I would approach this in layers. First, I'd implement frame-level analysis using a CNN to detect visual artifacts typical of deepfakes — things like inconsistent lighting, blending boundaries, and unnatural eye movements. Then I'd add temporal analysis to check for consistency across frames. In my experience, combining multiple weak signals gives much better results than any single detector. I have experience with this from my previous company where I built a multi-modal verification system."},
            
            {"time": 165, "action": "speak", "pid": "p1", "duration": 6,
             "text": "How do you handle the latency requirements for real-time processing?"},
            {"time": 175, "action": "speak", "pid": "p2", "duration": 40,
             "text": "In my current role, I designed a pipeline that processes video frames in under 50 milliseconds. The key was using model distillation to create lightweight models for initial screening, then only running the full analysis on flagged frames. I also implemented async processing with GPU batching, which improved throughput by 3x. I would take a similar approach here."},
            
            {"time": 220, "action": "speak", "pid": "p1", "duration": 8,
             "text": "Nice. Let me share my screen to walk through a coding problem."},
            {"time": 228, "action": "screen_share_start", "pid": "p1"},
            {"time": 230, "action": "speak", "pid": "p2", "duration": 35,
             "text": "I can see the problem. So we need to find the optimal way to match participants to streams. I would use a graph-based approach, creating a bipartite graph where participants are on one side and streams on the other. We can then use the Hungarian algorithm for optimal matching. Let me walk through the implementation."},
        ]
    
    def _scenario_device_name(self):
        """Candidate joins as 'MacBook Pro' instead of their name."""
        self.context = MeetingContext(
            candidate_name="Jordan Williams",
            candidate_email="jordan.w@outlook.com",
            interviewer_names=["Sarah Miller", "Tom Park"],
            scheduled_start_time=self.base_time,
            meeting_title="Interview - Jordan Williams - Backend",
            company_name="Sherlock",
            calendar_invite_participants=[
                {"name": "Sarah Miller", "email": "sarah@sherlock.ai", "role": "organizer"},
                {"name": "Tom Park", "email": "tom@sherlock.ai", "role": "interviewer"},
                {"name": "Jordan Williams", "email": "jordan.w@outlook.com", "role": "candidate"},
            ]
        )
        
        self.timeline = [
            {"time": -180, "action": "join", "pid": "p1", "name": "Sarah Miller"},
            {"time": -180, "action": "webcam_on", "pid": "p1"},
            {"time": -60, "action": "join", "pid": "p2", "name": "Tom Park"},
            {"time": -60, "action": "webcam_on", "pid": "p2"},
            {"time": 10, "action": "join", "pid": "p3", "name": "MacBook Pro"},
            {"time": 12, "action": "webcam_on", "pid": "p3"},
            
            {"time": 20, "action": "speak", "pid": "p1", "duration": 10,
             "text": "Hi! Are you Jordan? Looks like your name is showing as MacBook Pro. Can you hear us okay?"},
            {"time": 35, "action": "speak", "pid": "p3", "duration": 15,
             "text": "Oh sorry about that! Yes, I'm Jordan Williams. Let me fix my display name. I've been really looking forward to this interview, thanks for having me."},
            
            {"time": 55, "action": "speak", "pid": "p2", "duration": 8,
             "text": "No worries at all. I'm Tom, I lead the backend team. Let's get started. Tell me about your experience with distributed systems."},
            {"time": 68, "action": "speak", "pid": "p3", "duration": 40,
             "text": "Absolutely. At my previous company, I architected a microservices platform handling 50,000 requests per second. I designed the event-driven architecture using Kafka for message streaming. One challenge I solved was implementing exactly-once delivery semantics, which I accomplished using idempotency keys and a custom deduplication layer. I'm really excited about applying this experience at Sherlock."},
            
            {"time": 115, "action": "speak", "pid": "p1", "duration": 10,
             "text": "That's relevant to what we're building. How would you handle real-time data processing at scale?"},
            {"time": 130, "action": "speak", "pid": "p3", "duration": 35,
             "text": "I have extensive experience with real-time data pipelines. In my last role, I built a stream processing system that analyzed millions of events. I used Apache Flink for windowed aggregations and designed the system to handle backpressure gracefully. My approach would be to start with a solid event schema and build composable processing stages."},
            
            {"time": 170, "action": "speak", "pid": "p2", "duration": 6,
             "text": "Can you walk us through a time you had to debug a production issue?"},
            {"time": 180, "action": "speak", "pid": "p3", "duration": 45,
             "text": "One example that comes to mind: we had a cascading failure in our payment processing pipeline. The root cause was a memory leak in a Go service that only manifested under high load. I led the investigation using distributed tracing with Jaeger, identified the goroutine leak, and implemented a fix within 4 hours. After that, I set up automated canary deployments and memory profiling in CI to prevent similar issues."},
        ]
    
    def _scenario_nickname(self):
        """Candidate uses a nickname (Bill instead of William)."""
        self.context = MeetingContext(
            candidate_name="William Roberts",
            candidate_email="will.roberts@protonmail.com",
            interviewer_names=["Emily Zhang"],
            scheduled_start_time=self.base_time,
            meeting_title="Interview Round 2 - William Roberts",
            company_name="Sherlock",
            calendar_invite_participants=[
                {"name": "Emily Zhang", "email": "emily@sherlock.ai", "role": "organizer"},
                {"name": "William Roberts", "email": "will.roberts@protonmail.com", "role": "candidate"},
            ]
        )
        
        self.timeline = [
            {"time": -90, "action": "join", "pid": "p1", "name": "Emily Zhang"},
            {"time": -90, "action": "webcam_on", "pid": "p1"},
            {"time": 5, "action": "join", "pid": "p2", "name": "Bill Roberts"},
            {"time": 8, "action": "webcam_on", "pid": "p2"},
            
            {"time": 15, "action": "speak", "pid": "p1", "duration": 8,
             "text": "Hi Bill! Welcome back for round two. I'm Emily, I'll be doing the system design portion today."},
            {"time": 28, "action": "speak", "pid": "p2", "duration": 12,
             "text": "Hi Emily, great to meet you! Yeah, I go by Bill. I really enjoyed the first round and I've been prepping for this one."},
            
            {"time": 45, "action": "speak", "pid": "p1", "duration": 10,
             "text": "Let's dive in. How would you design a real-time notification system that handles millions of users?"},
            {"time": 60, "action": "speak", "pid": "p2", "duration": 55,
             "text": "Great question. I'd start by defining the requirements. For real-time at scale, I'd use a push-based architecture with WebSockets for connected clients and a fallback to long polling. The backend would use a pub-sub system like Redis Streams or Kafka. At my last company, I built something similar for our collaboration platform serving 2 million daily active users. The key insight was partitioning notification channels by user ID hash to distribute load evenly."},
            
            {"time": 120, "action": "speak", "pid": "p1", "duration": 6,
             "text": "How would you handle delivery guarantees?"},
            {"time": 130, "action": "speak", "pid": "p2", "duration": 40,
             "text": "I'd implement an at-least-once delivery model with client-side deduplication. Each notification gets a unique ID and the client tracks the last received ID. I'd also build a read receipt system so we know which notifications were actually seen versus just delivered. In my experience at my previous role, we used this pattern and achieved 99.97% delivery rate within 500 milliseconds."},
        ]
    
    def _scenario_wrong_name(self):
        """Interviewer entered the wrong candidate name in the system."""
        self.context = MeetingContext(
            candidate_name="Michael Chen",  # WRONG NAME! Actually is "Michelle Chen"
            candidate_email="m.chen@email.com",
            interviewer_names=["David Kumar"],
            scheduled_start_time=self.base_time,
            meeting_title="Technical Screen - Michael Chen",
            company_name="Sherlock",
            calendar_invite_participants=[
                {"name": "David Kumar", "email": "david@sherlock.ai", "role": "organizer"},
                {"name": "Michael Chen", "email": "m.chen@email.com", "role": "candidate"},
            ]
        )
        
        self.timeline = [
            {"time": -60, "action": "join", "pid": "p1", "name": "David Kumar"},
            {"time": -60, "action": "webcam_on", "pid": "p1"},
            {"time": 15, "action": "join", "pid": "p2", "name": "Michelle Chen"},
            {"time": 18, "action": "webcam_on", "pid": "p2"},
            
            {"time": 25, "action": "speak", "pid": "p1", "duration": 8,
             "text": "Hi! Are you Michelle? I have a Michael Chen on my calendar — must be a typo."},
            {"time": 38, "action": "speak", "pid": "p2", "duration": 10,
             "text": "Hi David! Yes, I'm Michelle Chen. Happens all the time with my name. Thanks for having me!"},
            
            {"time": 52, "action": "speak", "pid": "p1", "duration": 8,
             "text": "No problem. Tell me about yourself and your experience with ML systems."},
            {"time": 65, "action": "speak", "pid": "p2", "duration": 50,
             "text": "Sure! I'm a machine learning engineer with 5 years of experience. I graduated from Stanford with a Masters in CS focused on computer vision. At my current company, I lead the ML platform team where I've built and deployed over 20 production models. My expertise is in real-time inference systems, which I know is core to what Sherlock does. I designed a model serving platform that handles 100K predictions per second with p99 latency under 20ms."},
            
            {"time": 120, "action": "speak", "pid": "p1", "duration": 6,
             "text": "How do you approach model evaluation and monitoring in production?"},
            {"time": 130, "action": "speak", "pid": "p2", "duration": 40,
             "text": "I believe in comprehensive monitoring. In my experience, I set up automated drift detection using statistical tests on feature distributions. I also implement shadow scoring where new models run alongside production models for comparison. One system I built detected a data quality issue that would have caused a 15% accuracy drop if we hadn't caught it before full deployment."},
        ]
    
    def _scenario_panel_observers(self):
        """Panel interview with multiple interviewers and silent observers."""
        self.context = MeetingContext(
            candidate_name="Aisha Patel",
            candidate_email="aisha.p@yahoo.com",
            interviewer_names=["Mark Johnson", "Lisa Wang", "Ryan O'Brien"],
            scheduled_start_time=self.base_time,
            meeting_title="Final Round Panel - Aisha Patel",
            company_name="Sherlock",
            calendar_invite_participants=[
                {"name": "Mark Johnson", "email": "mark@sherlock.ai", "role": "organizer"},
                {"name": "Lisa Wang", "email": "lisa@sherlock.ai", "role": "interviewer"},
                {"name": "Ryan O'Brien", "email": "ryan@sherlock.ai", "role": "interviewer"},
                {"name": "Aisha Patel", "email": "aisha.p@yahoo.com", "role": "candidate"},
                {"name": "HR Bot", "email": "hr-recorder@sherlock.ai", "role": "observer"},
            ]
        )
        
        self.timeline = [
            {"time": -300, "action": "join", "pid": "p1", "name": "Mark Johnson"},
            {"time": -300, "action": "webcam_on", "pid": "p1"},
            {"time": -240, "action": "join", "pid": "p2", "name": "Lisa Wang"},
            {"time": -240, "action": "webcam_on", "pid": "p2"},
            {"time": -180, "action": "join", "pid": "p3", "name": "Ryan O'Brien"},
            {"time": -60, "action": "join", "pid": "p4", "name": "HR Recording Bot"},
            {"time": 5, "action": "join", "pid": "p5", "name": "Aisha Patel"},
            {"time": 8, "action": "webcam_on", "pid": "p5"},
            {"time": 30, "action": "join", "pid": "p6", "name": "Jennifer Liu"},
            
            {"time": 12, "action": "speak", "pid": "p1", "duration": 15,
             "text": "Hi Aisha, welcome to the final round! I'm Mark, the engineering manager. You've already met Lisa from the first round. And we have Ryan from the platform team joining us today."},
            {"time": 30, "action": "speak", "pid": "p5", "duration": 12,
             "text": "Hi everyone! Great to see you again Lisa. Nice to meet you Mark and Ryan. I'm Aisha, and I'm really excited about this final round."},
            
            {"time": 48, "action": "speak", "pid": "p2", "duration": 8,
             "text": "Great to see you again Aisha! Let's start with a system design question. How would you design a real-time fraud detection pipeline?"},
            {"time": 60, "action": "speak", "pid": "p5", "duration": 55,
             "text": "I'd design this as an event-driven architecture. Raw signals come in from the meeting platform — audio, video, and metadata streams. I'd process them through independent microservices, each running a specific detector. A fusion layer would combine the outputs using weighted scoring, similar to how credit card fraud detection works with ensemble models. In my previous role, I built exactly this kind of multi-signal system for detecting payment fraud. The key is making each detector independent so they can be updated and scaled individually."},
            
            {"time": 120, "action": "speak", "pid": "p3", "duration": 8,
             "text": "How would you handle false positives in such a system?"},
            {"time": 132, "action": "speak", "pid": "p5", "duration": 40,
             "text": "False positives are critical because flagging a legitimate candidate hurts user trust. I would implement a confidence threshold system where only alerts above a certain score trigger action. Below that, we'd flag for human review. I'd also build feedback loops where human reviewers' decisions feed back into model training. At my last company, this approach reduced false positives by 60% over six months while maintaining recall."},
            
            {"time": 178, "action": "speak", "pid": "p1", "duration": 6,
             "text": "What's your experience with team leadership and mentoring?"},
            {"time": 188, "action": "speak", "pid": "p5", "duration": 35,
             "text": "I've led teams of up to eight engineers. My approach is to empower team members with clear goals and context, then give them autonomy. I run weekly one-on-ones and quarterly growth conversations. I mentored three junior engineers to mid-level promotions over two years. I believe in leading by example — I still write code regularly and do code reviews to stay connected with the technical work."},
        ]
    
    def _scenario_name_change(self):
        """Candidate joins with generic name then changes it mid-meeting."""
        self.context = MeetingContext(
            candidate_name="Carlos Rodriguez",
            candidate_email="carlos.r@gmail.com",
            interviewer_names=["Nina Patel"],
            scheduled_start_time=self.base_time,
            meeting_title="Interview - Carlos Rodriguez",
            company_name="Sherlock",
            calendar_invite_participants=[
                {"name": "Nina Patel", "email": "nina@sherlock.ai", "role": "organizer"},
                {"name": "Carlos Rodriguez", "email": "carlos.r@gmail.com", "role": "candidate"},
            ]
        )
        
        self.timeline = [
            {"time": -60, "action": "join", "pid": "p1", "name": "Nina Patel"},
            {"time": -60, "action": "webcam_on", "pid": "p1"},
            {"time": 10, "action": "join", "pid": "p2", "name": "Guest User 42"},
            {"time": 15, "action": "webcam_on", "pid": "p2"},
            
            {"time": 20, "action": "speak", "pid": "p1", "duration": 6,
             "text": "Hi there! Is this Carlos? You're showing as Guest User."},
            {"time": 30, "action": "speak", "pid": "p2", "duration": 8,
             "text": "Oh yes, sorry! I'm Carlos Rodriguez. I'm joining from my roommate's computer. Let me change my name real quick."},
            
            {"time": 42, "action": "name_change", "pid": "p2", "new_name": "Carlos Rodriguez"},
            
            {"time": 48, "action": "speak", "pid": "p1", "duration": 8,
             "text": "No problem at all! I'm Nina, and I'll be your interviewer today. Tell me about your experience with web development."},
            {"time": 60, "action": "speak", "pid": "p2", "duration": 45,
             "text": "Of course! I've been a full-stack developer for 6 years. I started with front-end work in React and gradually moved to backend systems with Node and Python. In my current role, I architect and build APIs that serve our mobile and web apps. I've built RESTful and GraphQL services handling millions of requests. I'm particularly interested in Sherlock because of the real-time processing challenges."},
            
            {"time": 110, "action": "speak", "pid": "p1", "duration": 6,
             "text": "How do you approach testing and code quality?"},
            {"time": 120, "action": "speak", "pid": "p2", "duration": 35,
             "text": "I'm a strong advocate for testing. I follow a testing pyramid approach with comprehensive unit tests, integration tests for API boundaries, and selective end-to-end tests. In my previous role, I introduced property-based testing which caught edge cases we never would have found manually. I also implemented automated code review checks and maintained 85% code coverage across our services."},
        ]
    
    def _scenario_ambiguous(self):
        """Deliberately ambiguous scenario - stress test for the system."""
        self.context = MeetingContext(
            candidate_name="Sam Taylor",
            candidate_email="s.taylor@mail.com",
            interviewer_names=["Pat Taylor"],  # Same last name!
            scheduled_start_time=self.base_time,
            meeting_title="Interview",
            company_name="Sherlock",
            calendar_invite_participants=[
                {"name": "Pat Taylor", "email": "pat@sherlock.ai", "role": "organizer"},
                {"name": "Sam Taylor", "email": "s.taylor@mail.com", "role": "candidate"},
            ]
        )
        
        self.timeline = [
            {"time": -30, "action": "join", "pid": "p1", "name": "Pat Taylor"},
            {"time": 5, "action": "join", "pid": "p2", "name": "S. Taylor"},
            {"time": 8, "action": "webcam_on", "pid": "p2"},
            
            {"time": 15, "action": "speak", "pid": "p1", "duration": 8,
             "text": "Hey there! I'm Pat Taylor from the engineering team. Are you Sam?"},
            {"time": 28, "action": "speak", "pid": "p2", "duration": 10,
             "text": "Hi Pat! Yes, I'm Sam Taylor. Funny that we share the last name! I'm really excited about this role at Sherlock."},
            
            {"time": 42, "action": "speak", "pid": "p1", "duration": 8,
             "text": "Ha, yeah! Let me tell you about the team and then we'll dive into some technical questions. We work on real-time fraud detection here."},
            {"time": 55, "action": "speak", "pid": "p2", "duration": 45,
             "text": "That sounds fascinating. In my background, I've worked extensively on real-time systems. At my last company, I was the tech lead for our anomaly detection platform. I built models that processed streaming data and flagged suspicious patterns within milliseconds. My thesis was actually on online learning algorithms, so I have deep expertise in systems that adapt in real-time."},
            
            {"time": 105, "action": "speak", "pid": "p1", "duration": 6,
             "text": "Tell me about a challenging technical problem you solved recently."},
            {"time": 115, "action": "speak", "pid": "p2", "duration": 40,
             "text": "One significant challenge was building a feature store for our ML pipeline. We needed sub-millisecond reads for online inference while also supporting batch training workflows. I designed a dual-storage architecture using Redis for real-time serving and Parquet files on S3 for batch. The system I implemented reduced feature computation time by 80% and eliminated feature skew between training and serving."},
            
            {"time": 160, "action": "webcam_on", "pid": "p1"},
        ]
    
    def get_phases(self):
        """
        Return the timeline as progressive phases for the demo.
        Each phase reveals more events that the engine processes.
        """
        if not self.timeline:
            return []
        
        # Group events into phases
        phases = []
        current_phase = {"name": "Meeting Setup", "events": [], "time_range": []}
        
        for item in sorted(self.timeline, key=lambda x: x["time"]):
            t = self.base_time + item["time"]
            
            if item["time"] < 0:
                phase_name = "Pre-Meeting"
            elif item["time"] < 30:
                phase_name = "Introductions"
            elif item["time"] < 100:
                phase_name = "Early Interview"
            else:
                phase_name = "Deep Interview"
            
            if phase_name != current_phase["name"]:
                if current_phase["events"]:
                    phases.append(current_phase)
                current_phase = {"name": phase_name, "events": [], "time_range": []}
            
            current_phase["events"].append(item)
            current_phase["time_range"].append(t)
        
        if current_phase["events"]:
            phases.append(current_phase)
        
        return phases
    
    def process_event_item(self, item, engine):
        """
        Process a single timeline item, creating the appropriate
        model objects and feeding them to the engine.
        """
        t = self.base_time + item["time"]
        pid = item["pid"]
        action = item["action"]
        
        if action == "join":
            participant = Participant(
                participant_id=pid,
                display_name=item["name"],
                join_time=t,
                is_active=True,
            )
            engine.add_participant(participant)
            engine.add_event(_make_event(t, pid, EventType.JOIN))
        
        elif action == "leave":
            engine.add_event(_make_event(t, pid, EventType.LEAVE))
        
        elif action == "webcam_on":
            engine.add_event(_make_event(t, pid, EventType.WEBCAM_ON))
        
        elif action == "webcam_off":
            engine.add_event(_make_event(t, pid, EventType.WEBCAM_OFF))
        
        elif action == "screen_share_start":
            engine.add_event(_make_event(t, pid, EventType.SCREEN_SHARE_START))
        
        elif action == "screen_share_stop":
            engine.add_event(_make_event(t, pid, EventType.SCREEN_SHARE_STOP))
        
        elif action == "name_change":
            engine.add_event(_make_event(t, pid, EventType.NAME_CHANGE, 
                                          {"new_name": item["new_name"]}))
        
        elif action == "speak":
            duration = item.get("duration", 10)
            text = item.get("text", "")
            
            segment = SpeakingSegment(
                participant_id=pid,
                start_time=t,
                end_time=t + duration,
                transcript_text=text,
            )
            engine.add_speaking_segment(segment)
            
            if text:
                entry = TranscriptEntry(
                    timestamp=t,
                    participant_id=pid,
                    text=text,
                )
                engine.add_transcript_entry(entry)
    
    def run_full(self, engine):
        """Run all events through the engine and return results at each phase."""
        phases = self.get_phases()
        results = []
        
        for phase in phases:
            for item in phase["events"]:
                self.process_event_item(item, engine)
            
            result = engine.identify()
            results.append({
                "phase": phase["name"],
                "result": result,
            })
        
        return results


# Available scenarios for the demo
SCENARIOS = {
    "standard": {
        "name": "Standard Interview",
        "description": "Normal 1-on-1 interview where candidate name matches",
        "difficulty": "Easy",
    },
    "device_name": {
        "name": "Device Name",
        "description": "Candidate joins as 'MacBook Pro' instead of their name",
        "difficulty": "Medium",
    },
    "nickname": {
        "name": "Nickname",
        "description": "Candidate uses 'Bill' instead of 'William'",
        "difficulty": "Medium",
    },
    "wrong_name": {
        "name": "Wrong Name in System",
        "description": "System has 'Michael' but candidate is actually 'Michelle'",
        "difficulty": "Hard",
    },
    "panel_with_observers": {
        "name": "Panel + Observers",
        "description": "3 interviewers, 1 recording bot, and a late-joining observer",
        "difficulty": "Hard",
    },
    "name_change": {
        "name": "Name Change Mid-Meeting",
        "description": "Candidate joins as 'Guest User 42' then changes name",
        "difficulty": "Medium",
    },
    "ambiguous": {
        "name": "Ambiguous (Stress Test)",
        "description": "Similar names, minimal metadata, deliberately tricky",
        "difficulty": "Very Hard",
    },
}
