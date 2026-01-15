"""
Flask server for MASE interactive dialogue interface.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Provides REST API and Server-Sent Events for real-time
dialogue streaming with human participation.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import asdict

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# Handle both package and direct execution
try:
    from .ollama_client import OllamaClient
    from .agents import EnsembleConfig, AgentLoader
    from .interactive_orchestrator import (
        InteractiveSession,
        SessionState,
        TurnEvent,
        StateEvent,
        create_interactive_session
    )
    from .session_analysis import analyze_session
except ImportError:
    from ollama_client import OllamaClient
    from agents import EnsembleConfig, AgentLoader
    from interactive_orchestrator import (
        InteractiveSession,
        SessionState,
        TurnEvent,
        StateEvent,
        create_interactive_session
    )
    from session_analysis import analyze_session

# Flask app
app = Flask(__name__, static_folder='../web', static_url_path='')
CORS(app)

# Active sessions store
sessions: Dict[str, InteractiveSession] = {}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
AGENTS_DIR = PROJECT_ROOT / ".claude" / "agents"
CONFIG_DIR = PROJECT_ROOT / "experiments" / "config"
SESSIONS_DIR = PROJECT_ROOT / "sessions"

# Agent color mapping
AGENT_COLORS = {
    "luma": "#F59E0B",    # Yellow
    "elowen": "#10B981",  # Green
    "orin": "#3B82F6",    # Blue
    "nyra": "#8B5CF6",    # Purple
    "ilya": "#06B6D4",    # Cyan
    "sefi": "#F97316",    # Orange
    "tala": "#EF4444",    # Red
    "human": "#E5E7EB"    # Light gray
}

# Agent descriptions
AGENT_DESCRIPTIONS = {
    "luma": "Child voice, moral clarity",
    "elowen": "Ecological wisdom, kincentric",
    "orin": "Systems thinking, cybernetics",
    "nyra": "Moral imagination, design fiction",
    "ilya": "Liminal guide, posthuman",
    "sefi": "Policy pragmatist, governance",
    "tala": "Capitalist realist, markets"
}


# ============================================================================
# Static file serving
# ============================================================================

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')


# ============================================================================
# API: System status
# ============================================================================

@app.route('/api/status')
def get_status():
    """Get system status including Ollama availability."""
    ollama_running = OllamaClient.is_running()
    models = []

    if ollama_running:
        try:
            models = OllamaClient.get_available_models()
        except Exception:
            pass

    return jsonify({
        "ollama_running": ollama_running,
        "available_models": models,
        "active_sessions": len(sessions)
    })


# ============================================================================
# API: Agents
# ============================================================================

@app.route('/api/agents')
def get_agents():
    """Get list of all agents with metadata."""
    agents = []

    # Load agent definitions
    try:
        loader = AgentLoader(AGENTS_DIR)
        agent_defs = loader.load_all()

        for agent_id, agent in agent_defs.items():
            # Use agent_id capitalized as display name
            display_name = agent_id.capitalize()
            agents.append({
                "id": agent_id,
                "name": display_name,
                "full_name": agent.name,
                "color": AGENT_COLORS.get(agent_id, "#888888"),
                "description": AGENT_DESCRIPTIONS.get(agent_id, ""),
                "is_human": False
            })
    except Exception as e:
        # Fallback to basic list
        for agent_id in AGENT_COLORS:
            if agent_id != "human":
                agents.append({
                    "id": agent_id,
                    "name": agent_id.capitalize(),
                    "full_name": f"{agent_id}-agent",
                    "color": AGENT_COLORS.get(agent_id, "#888888"),
                    "description": AGENT_DESCRIPTIONS.get(agent_id, ""),
                    "is_human": False
                })

    # Add human
    agents.append({
        "id": "human",
        "name": "You",
        "full_name": "human-participant",
        "color": AGENT_COLORS["human"],
        "description": "Human participant",
        "is_human": True
    })

    return jsonify({"agents": agents})


# ============================================================================
# API: Session management
# ============================================================================

@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new dialogue session."""
    data = request.get_json() or {}
    provocation = data.get('provocation', 'What does it mean to live well?')
    seed = data.get('seed', 42)
    config_name = data.get('config', 'multi_model')

    # Check Ollama
    if not OllamaClient.is_running():
        return jsonify({"error": "Ollama is not running"}), 503

    # Load config
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        return jsonify({"error": f"Config not found: {config_name}"}), 400

    try:
        config = EnsembleConfig.from_yaml(config_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load config: {e}"}), 500

    # Create session
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    session = InteractiveSession(
        config=config,
        provocation=provocation,
        output_dir=SESSIONS_DIR,
        seed=seed,
        agents_dir=AGENTS_DIR
    )

    # Store session
    sessions[session.session_id] = session

    return jsonify({
        "session_id": session.session_id,
        "provocation": provocation,
        "agents": session.get_agents_metadata()
    })


@app.route('/api/session/<session_id>/state')
def get_session_state(session_id: str):
    """Get current state of a session."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    state = session.get_state()

    # Add history
    history = []
    for agent_id, agent_name, content in session.dialogue_history:
        name = "You" if agent_id == "human" else (
            agent_name.split('-')[0].capitalize() if agent_name else agent_id
        )
        history.append({
            "agent_id": agent_id,
            "name": name,
            "content": content,
            "color": AGENT_COLORS.get(agent_id, "#888888")
        })

    state["history"] = history
    state["provocation"] = session.provocation

    return jsonify(state)


@app.route('/api/session/<session_id>/stream')
def stream_session(session_id: str):
    """SSE endpoint for streaming dialogue turns.

    Queue-based architecture:
    - Session runs in background thread, pushes events to queue
    - This endpoint reads from queue and sends SSE events
    - Reconnection-safe: queue persists across SSE connections
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]

    # Start the session if not already started (idempotent)
    session.start()

    def generate():
        """Generator that reads from session's event queue."""
        try:
            while True:
                # Check if session is complete
                if session.state == SessionState.COMPLETE:
                    # Drain any remaining events
                    while session.has_events():
                        event = session.get_next_event(timeout=0.1)
                        if event:
                            yield format_sse_event(event)
                    break

                # Get next event with timeout (allows periodic checking)
                event = session.get_next_event(timeout=5.0)

                if event is None:
                    # Timeout - send keepalive comment
                    yield ": keepalive\n\n"
                    continue

                yield format_sse_event(event)

                # If session just completed, send final event and exit
                if isinstance(event, StateEvent) and event.state == SessionState.COMPLETE:
                    break

        except GeneratorExit:
            # Client disconnected - that's fine, queue persists
            pass
        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


def format_sse_event(event) -> str:
    """Format a TurnEvent or StateEvent as SSE data."""
    if isinstance(event, TurnEvent):
        data = {
            "type": "turn",
            "turn_number": event.turn_number,
            "agent_id": event.agent_id,
            "agent_name": event.agent_name,
            "content": event.content,
            "model": event.model,
            "latency_ms": event.latency_ms,
            "is_human": event.is_human,
            "color": AGENT_COLORS.get(event.agent_id, "#888888")
        }
        return f"event: turn\ndata: {json.dumps(data)}\n\n"

    elif isinstance(event, StateEvent):
        data = {
            "type": "state",
            "state": event.state.value,
            "next_speaker": event.next_speaker,
            "message": event.message
        }
        return f"event: state\ndata: {json.dumps(data)}\n\n"

    return ""


@app.route('/api/session/<session_id>/pause', methods=['POST'])
def pause_session(session_id: str):
    """Pause an active session."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    session.pause()

    return jsonify({"status": "paused"})


@app.route('/api/session/<session_id>/resume', methods=['POST'])
def resume_session(session_id: str):
    """Resume a paused session."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    session.resume()

    return jsonify({"status": "resumed"})


@app.route('/api/session/<session_id>/human', methods=['POST'])
def submit_human_turn(session_id: str):
    """Submit human's contribution to the dialogue."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json() or {}
    content = data.get('content', '').strip()

    if not content:
        return jsonify({"error": "Content is required"}), 400

    session = sessions[session_id]

    # Submit the turn
    turn_event = session.submit_human_turn(content)

    return jsonify({
        "status": "submitted",
        "turn_number": turn_event.turn_number,
        "content": turn_event.content
    })


@app.route('/api/session/<session_id>/invoke', methods=['POST'])
def invoke_agent(session_id: str):
    """Request a specific agent to speak next."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json() or {}
    agent_id = data.get('agent_id', '').strip()

    if not agent_id:
        return jsonify({"error": "agent_id is required"}), 400

    session = sessions[session_id]
    session.invoke_agent(agent_id)

    return jsonify({"status": "invoked", "agent_id": agent_id})


@app.route('/api/session/<session_id>/end', methods=['POST'])
def end_session_endpoint(session_id: str):
    """End a session, run analysis, and save results."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    path = session.end_session()

    # Clean up session from store
    del sessions[session_id]

    # Run post-hoc analysis if session was saved
    analysis_result = None
    analysis_path = None

    if path and path.exists():
        try:
            result = analyze_session(path, compute_embeddings=True)
            analysis_result = result.to_dict()

            # Save analysis to separate file
            analysis_path = path.with_name(
                path.stem.replace('_checkpoint', '_analysis') + '.json'
            )
            with open(analysis_path, 'w') as f:
                json.dump(analysis_result, f, indent=2)

        except Exception as e:
            print(f"Analysis failed: {e}")
            analysis_result = {"error": str(e)}

    return jsonify({
        "status": "ended",
        "saved_to": str(path) if path else None,
        "analysis_path": str(analysis_path) if analysis_path else None,
        "analysis": analysis_result
    })


# ============================================================================
# Session History & Analysis Endpoints
# ============================================================================

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all saved sessions with analysis status."""
    sessions_list = []

    for checkpoint in sorted(SESSIONS_DIR.glob('*_checkpoint.json'), reverse=True):
        # Extract just the timestamp part (e.g., "20260115_141851" from "session_20260115_141851_checkpoint")
        session_id = checkpoint.stem.replace('_checkpoint', '').replace('session_', '')
        analysis_path = checkpoint.with_name(
            checkpoint.stem.replace('_checkpoint', '_analysis') + '.json'
        )

        # Get basic info from checkpoint
        try:
            with open(checkpoint) as f:
                data = json.load(f)
                provocation = data.get('provocation', '')[:100]
                n_turns = len(data.get('turns', []))
                timestamp = data.get('start_time', '')
        except Exception:
            provocation = ''
            n_turns = 0
            timestamp = ''

        sessions_list.append({
            'session_id': session_id,
            'checkpoint_path': str(checkpoint),
            'has_analysis': analysis_path.exists(),
            'analysis_path': str(analysis_path) if analysis_path.exists() else None,
            'provocation': provocation,
            'n_turns': n_turns,
            'timestamp': timestamp
        })

    return jsonify({'sessions': sessions_list})


@app.route('/api/sessions/<session_id>/analysis', methods=['GET'])
def get_session_analysis(session_id: str):
    """Get analysis for a specific session."""
    # Find the analysis file
    analysis_path = SESSIONS_DIR / f'session_{session_id}_analysis.json'

    if not analysis_path.exists():
        # Try to run analysis on the checkpoint
        checkpoint_path = SESSIONS_DIR / f'session_{session_id}_checkpoint.json'
        if not checkpoint_path.exists():
            return jsonify({"error": "Session not found"}), 404

        try:
            result = analyze_session(checkpoint_path, compute_embeddings=True)
            analysis_result = result.to_dict()

            # Save for future requests
            with open(analysis_path, 'w') as f:
                json.dump(analysis_result, f, indent=2)

            return jsonify(analysis_result)

        except Exception as e:
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    # Load existing analysis
    with open(analysis_path) as f:
        return jsonify(json.load(f))


@app.route('/api/sessions/<session_id>/dialogue', methods=['GET'])
def get_session_dialogue(session_id: str):
    """Get full dialogue content for a session."""
    checkpoint_path = SESSIONS_DIR / f'session_{session_id}_checkpoint.json'

    if not checkpoint_path.exists():
        return jsonify({"error": "Session not found"}), 404

    with open(checkpoint_path) as f:
        data = json.load(f)

    return jsonify({
        'session_id': session_id,
        'provocation': data.get('provocation', ''),
        'turns': data.get('turns', []),
        'start_time': data.get('start_time', ''),
        'end_time': data.get('end_time', '')
    })


# ============================================================================
# Main
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 5050, debug: bool = False):
    """Run the Flask server."""
    print(f"\n{'='*60}")
    print("MASE Interactive Dialogue Server")
    print(f"{'='*60}")
    print(f"Server: http://{host}:{port}")
    print(f"Agents: {AGENTS_DIR}")
    print(f"Sessions: {SESSIONS_DIR}")
    print(f"{'='*60}\n")

    # Check Ollama
    if OllamaClient.is_running():
        models = OllamaClient.get_available_models()
        print(f"Ollama: Running ({len(models)} models available)")
    else:
        print("WARNING: Ollama is not running!")
        print("Start with: ollama serve")

    print()

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_server(debug=True)
