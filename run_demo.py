"""
Sherlock Candidate Identification System — Demo Runner

Entry point for running the interactive demo. Supports:
1. Web dashboard mode (default) - launches Flask web server
2. CLI mode - runs scenarios in the terminal with formatted output
"""

import sys
import os

# Reconfigure stdout and stderr to use UTF-8 to prevent encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_cli_demo(scenario_name="standard"):
    """Run a scenario in CLI mode with formatted output."""
    from sherlock.engine import CandidateIdentificationEngine
    from sherlock.simulator import MeetingSimulator, SCENARIOS
    
    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        return
    
    info = SCENARIOS[scenario_name]
    print("\n" + "=" * 70)
    print(f"  🔍 SHERLOCK CANDIDATE IDENTIFICATION SYSTEM")
    print(f"  Scenario: {info['name']} ({info['difficulty']})")
    print(f"  {info['description']}")
    print("=" * 70)
    
    sim = MeetingSimulator(scenario_name)
    engine = CandidateIdentificationEngine(sim.context)
    
    print(f"\n  📋 Meeting Context:")
    print(f"     Candidate Name: {sim.context.candidate_name}")
    print(f"     Candidate Email: {sim.context.candidate_email}")
    print(f"     Interviewers: {', '.join(sim.context.interviewer_names)}")
    print(f"     Meeting: {sim.context.meeting_title}")
    
    results = sim.run_full(engine)
    
    for phase_data in results:
        phase = phase_data["phase"]
        result = phase_data["result"]
        
        print(f"\n  {'─' * 60}")
        print(f"  ⏱  Phase: {phase}")
        print(f"  {'─' * 60}")
        
        print(f"\n  Status: {result.status.upper()} | Confidence: {result.overall_confidence:.0%}")
        print(f"  {result.explanation}")
        
        print(f"\n  {'Participant':<25} {'Probability':>12} {'Signals':>8}")
        print(f"  {'─' * 48}")
        
        for assessment in result.assessments:
            marker = " ✅" if assessment.is_identified else ""
            prob = f"{assessment.probability:.0%}"
            n_signals = len([s for s in assessment.signal_scores.values() if s.confidence > 0.1])
            
            print(f"  {assessment.display_name:<25} {prob:>12} {n_signals:>6}/6{marker}")
        
        # Show top reasons for top candidate
        top = result.get_top_candidate()
        if top and top.explanations:
            print(f"\n  💡 Why '{top.display_name}':")
            for exp in top.explanations[:3]:
                print(f"     • {exp}")
    
    # Final summary
    final = results[-1]["result"]
    top = final.get_top_candidate()
    
    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULT")
    print(f"{'=' * 70}")
    
    if top:
        status_icon = "✅" if final.status == "identified" else "🟡" if final.status == "likely" else "❓"
        print(f"  {status_icon} {top.display_name} — {top.probability:.0%} confidence")
        print(f"  Status: {final.status}")
        print(f"  {final.explanation}")
    else:
        print(f"  ❓ No candidate identified")
    
    print(f"\n{'=' * 70}\n")


def run_all_scenarios():
    """Run all scenarios and print a summary table."""
    from sherlock.engine import CandidateIdentificationEngine
    from sherlock.simulator import MeetingSimulator, SCENARIOS
    
    print("\n" + "=" * 80)
    print("  🔍 SHERLOCK — ALL SCENARIOS EVALUATION")
    print("=" * 80)
    
    results_summary = []
    
    for name, info in SCENARIOS.items():
        sim = MeetingSimulator(name)
        engine = CandidateIdentificationEngine(sim.context)
        results = sim.run_full(engine)
        final = results[-1]["result"]
        top = final.get_top_candidate()
        
        results_summary.append({
            "scenario": info["name"],
            "difficulty": info["difficulty"],
            "status": final.status,
            "confidence": final.overall_confidence,
            "candidate": top.display_name if top else "N/A",
            "correct": final.status in ("identified", "likely"),
        })
    
    print(f"\n  {'Scenario':<28} {'Difficulty':<12} {'Status':<12} {'Confidence':>10} {'Result':<8}")
    print(f"  {'─' * 75}")
    
    for r in results_summary:
        icon = "✅" if r["correct"] else "❌"
        print(f"  {r['scenario']:<28} {r['difficulty']:<12} {r['status']:<12} {r['confidence']:>9.0%} {icon}")
    
    correct = sum(1 for r in results_summary if r["correct"])
    total = len(results_summary)
    print(f"\n  Overall: {correct}/{total} scenarios correctly identified ({correct/total:.0%})")
    print(f"{'=' * 80}\n")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "cli":
            scenario = sys.argv[2] if len(sys.argv) > 2 else "standard"
            run_cli_demo(scenario)
        elif command == "all":
            run_all_scenarios()
        elif command == "web":
            from web.app import run
            run()
        else:
            print(f"Unknown command: {command}")
            print("Usage:")
            print("  python run_demo.py          # Launch web dashboard")
            print("  python run_demo.py web       # Launch web dashboard")
            print("  python run_demo.py cli       # Run standard scenario in CLI")
            print("  python run_demo.py cli <scenario>  # Run specific scenario")
            print("  python run_demo.py all       # Run all scenarios (evaluation)")
    else:
        # Default: launch web dashboard
        from web.app import run
        run()


if __name__ == "__main__":
    main()
