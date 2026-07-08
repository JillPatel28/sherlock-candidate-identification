"""
Sherlock Web Dashboard

Flask-based web application providing:
- Real-time candidate identification dashboard
- Interactive scenario selection and simulation
- Live confidence visualization
- Signal breakdown per participant
- Detailed explanation panel
"""

import json
import time
import logging
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from sherlock.engine import CandidateIdentificationEngine
from sherlock.simulator import MeetingSimulator, SCENARIOS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder="templates",
            static_folder="static")
app.config["SECRET_KEY"] = "sherlock-demo-key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Global state for the current simulation
current_engine = None
current_simulator = None
simulation_results = []
current_phase_index = 0


def _serialize_result(result):
    """Convert IdentificationResult to JSON-serializable dict."""
    assessments = []
    for a in result.assessments:
        signals = {}
        for name, s in a.signal_scores.items():
            signals[name] = {
                "score": round(s.score, 4),
                "confidence": round(s.confidence, 4),
                "explanation": s.explanation,
                "sub_scores": {k: round(v, 4) if isinstance(v, float) else v 
                               for k, v in s.sub_scores.items()},
            }
        
        assessments.append({
            "participant_id": a.participant_id,
            "display_name": a.display_name,
            "probability": round(a.probability, 4),
            "signal_scores": signals,
            "explanations": a.explanations[:5],
            "is_identified": a.is_identified,
        })
    
    return {
        "timestamp": result.timestamp,
        "assessments": assessments,
        "identified_candidate_id": result.identified_candidate_id,
        "overall_confidence": round(result.overall_confidence, 4),
        "status": result.status,
        "explanation": result.explanation,
    }


@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("dashboard.html", scenarios=SCENARIOS)


@app.route("/api/scenarios")
def get_scenarios():
    """List available demo scenarios."""
    return jsonify(SCENARIOS)


@app.route("/api/start", methods=["POST"])
def start_simulation():
    """Start a new simulation with the selected scenario."""
    global current_engine, current_simulator, simulation_results, current_phase_index
    
    data = request.get_json()
    scenario_name = data.get("scenario", "standard")
    
    if scenario_name not in SCENARIOS:
        return jsonify({"error": f"Unknown scenario: {scenario_name}"}), 400
    
    # Create simulator and engine
    current_simulator = MeetingSimulator(scenario_name)
    current_engine = CandidateIdentificationEngine(current_simulator.context)
    
    # Run all phases and store results
    simulation_results = current_simulator.run_full(current_engine)
    current_phase_index = 0
    
    # Return context info
    ctx = current_simulator.context
    context_info = {
        "candidate_name": ctx.candidate_name,
        "candidate_email": ctx.candidate_email,
        "interviewer_names": ctx.interviewer_names,
        "meeting_title": ctx.meeting_title,
        "total_phases": len(simulation_results),
        "scenario": SCENARIOS[scenario_name],
        "weights": current_engine.signal_weights,
    }
    
    return jsonify({"status": "ok", "context": context_info})


@app.route("/api/next-phase", methods=["POST"])
def next_phase():
    """Advance to the next phase and return updated results."""
    global current_phase_index
    
    if not simulation_results:
        return jsonify({"error": "No simulation running"}), 400
    
    if current_phase_index >= len(simulation_results):
        return jsonify({"error": "Simulation complete", "complete": True}), 200
    
    phase_data = simulation_results[current_phase_index]
    result = _serialize_result(phase_data["result"])
    result["phase_name"] = phase_data["phase"]
    result["phase_index"] = current_phase_index
    result["total_phases"] = len(simulation_results)
    
    current_phase_index += 1
    
    return jsonify(result)


@app.route("/api/all-phases", methods=["POST"])
def all_phases():
    """Return all phase results at once."""
    if not simulation_results:
        return jsonify({"error": "No simulation running"}), 400
    
    phases = []
    for i, phase_data in enumerate(simulation_results):
        result = _serialize_result(phase_data["result"])
        result["phase_name"] = phase_data["phase"]
        result["phase_index"] = i
        phases.append(result)
    
    return jsonify({"phases": phases})


@app.route("/api/signal-breakdown/<participant_id>")
def signal_breakdown(participant_id):
    """Get detailed signal breakdown for a participant."""
    if not current_engine:
        return jsonify({"error": "No simulation running"}), 400
    
    breakdown = current_engine.get_signal_breakdown(participant_id)
    return jsonify(breakdown)


@app.route("/api/confidence-history")
def confidence_history():
    """Get the confidence score history."""
    if not current_engine:
        return jsonify({"error": "No simulation running"}), 400
    
    history = current_engine.get_confidence_history()
    return jsonify(history)


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Receive confirmation feedback and adjust weights dynamically."""
    global current_engine
    
    if not current_engine:
        return jsonify({"error": "No simulation running"}), 400
        
    data = request.get_json()
    confirmed_id = data.get("participant_id")
    
    if not confirmed_id:
        return jsonify({"error": "Missing participant_id"}), 400
        
    if confirmed_id not in current_engine.participants:
        return jsonify({"error": "Participant not found in meeting"}), 400
        
    # Run learning process
    new_weights = current_engine.learn_from_feedback(confirmed_id)
    
    # Re-run engine identify to recalculate probabilities using new weights
    updated_result = current_engine.identify()
    serialized_result = _serialize_result(updated_result)
    
    return jsonify({
        "status": "ok",
        "weights": new_weights,
        "updated_result": serialized_result
    })


@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")
    emit("connected", {"status": "ok"})


@socketio.on("start_simulation")
def handle_start_simulation(data):
    """Start simulation via WebSocket."""
    global current_engine, current_simulator, simulation_results, current_phase_index
    
    scenario_name = data.get("scenario", "standard")
    
    current_simulator = MeetingSimulator(scenario_name)
    current_engine = CandidateIdentificationEngine(current_simulator.context)
    simulation_results = current_simulator.run_full(current_engine)
    current_phase_index = 0
    
    ctx = current_simulator.context
    emit("simulation_started", {
        "candidate_name": ctx.candidate_name,
        "candidate_email": ctx.candidate_email,
        "interviewer_names": ctx.interviewer_names,
        "meeting_title": ctx.meeting_title,
        "total_phases": len(simulation_results),
        "scenario": SCENARIOS[scenario_name],
    })


@socketio.on("request_next_phase")
def handle_next_phase():
    """Send next phase via WebSocket."""
    global current_phase_index
    
    if not simulation_results:
        emit("error", {"message": "No simulation running"})
        return
    
    if current_phase_index >= len(simulation_results):
        emit("simulation_complete", {"message": "All phases processed"})
        return
    
    phase_data = simulation_results[current_phase_index]
    result = _serialize_result(phase_data["result"])
    result["phase_name"] = phase_data["phase"]
    result["phase_index"] = current_phase_index
    result["total_phases"] = len(simulation_results)
    
    current_phase_index += 1
    
    emit("phase_update", result)


def run():
    """Start the web server."""
    print("\n" + "="*60)
    print("  🔍 Sherlock Candidate Identification System")
    print("  Dashboard: http://localhost:5000")
    print("="*60 + "\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    run()
