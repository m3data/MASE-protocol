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
    from .agents import (
        EnsembleConfig, AgentLoader, PersonaLoader, TemplateLoader,
        load_personas, Persona
    )
    from .interactive_orchestrator import (
        InteractiveSession,
        SessionState,
        TurnEvent,
        StateEvent,
        MetricsEvent,
        create_interactive_session
    )
    from .session_analysis import analyze_session
except ImportError:
    from ollama_client import OllamaClient
    from agents import (
        EnsembleConfig, AgentLoader, PersonaLoader, TemplateLoader,
        load_personas, Persona
    )
    from interactive_orchestrator import (
        InteractiveSession,
        SessionState,
        TurnEvent,
        StateEvent,
        MetricsEvent,
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
AGENTS_DIR = PROJECT_ROOT / "agents" / "personas"
TEMPLATES_DIR = PROJECT_ROOT / "agents" / "templates"
CONFIG_DIR = PROJECT_ROOT / "experiments" / "config"
SESSIONS_DIR = PROJECT_ROOT / "sessions"

# Persona and template loaders (cached)
_persona_loader: Optional[PersonaLoader] = None
_template_loader: Optional[TemplateLoader] = None


def get_persona_loader() -> PersonaLoader:
    """Get cached persona loader."""
    global _persona_loader
    if _persona_loader is None:
        _persona_loader = PersonaLoader(AGENTS_DIR, TEMPLATES_DIR)
        _persona_loader.load_all()
    return _persona_loader


def get_template_loader() -> TemplateLoader:
    """Get cached template loader."""
    global _template_loader
    if _template_loader is None:
        _template_loader = TemplateLoader(TEMPLATES_DIR)
        _template_loader.load_all()
    return _template_loader


def get_persona_color(persona_id: str) -> str:
    """Get color for a persona from YAML definition."""
    loader = get_persona_loader()
    persona = loader.get(persona_id)
    if persona:
        return persona.color
    # Fallback for human and researcher
    if persona_id == "human":
        return "#B49070"
    if persona_id == "researcher":
        return "#A0A0B4"
    return "#888888"


def get_persona_description(persona_id: str) -> str:
    """Get description for a persona from YAML definition."""
    loader = get_persona_loader()
    persona = loader.get(persona_id)
    if persona:
        return persona.description
    if persona_id == "human":
        return "Human participant"
    if persona_id == "researcher":
        return "Researcher interjection"
    return ""


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
# API: Templates
# ============================================================================

@app.route('/api/templates')
def get_templates():
    """Get list of all templates."""
    loader = get_template_loader()
    templates = loader.list_all()
    return jsonify({"templates": templates})


@app.route('/api/templates/<template_id>')
def get_template(template_id: str):
    """Get a specific template with full details."""
    loader = get_template_loader()
    template = loader.get(template_id)

    if not template:
        return jsonify({"error": "Template not found"}), 404

    return jsonify({
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "epistemic_lens": template.epistemic_lens,
        "voice_guidance": {
            "style": template.voice_guidance.style,
            "register": template.voice_guidance.register,
            "patterns": template.voice_guidance.patterns,
            "avoid": template.voice_guidance.avoid
        },
        "default_personality": {
            "openness": template.default_personality.openness,
            "conscientiousness": template.default_personality.conscientiousness,
            "extraversion": template.default_personality.extraversion,
            "agreeableness": template.default_personality.agreeableness,
            "neuroticism": template.default_personality.neuroticism
        }
    })


# ============================================================================
# API: Personas
# ============================================================================

@app.route('/api/personas')
def get_personas():
    """Get list of all personas with metadata."""
    loader = get_persona_loader()
    personas = loader.list_all()
    return jsonify({"personas": personas})


@app.route('/api/personas/<persona_id>')
def get_persona(persona_id: str):
    """Get a specific persona with full details including resolved template."""
    loader = get_persona_loader()
    persona = loader.get(persona_id)

    if not persona:
        return jsonify({"error": "Persona not found"}), 404

    result = {
        "id": persona.id,
        "name": persona.name,
        "template_id": persona.template_id,
        "description": persona.description,
        "color": persona.color,
        "character": persona.character,
        "signature_phrases": persona.signature_phrases,
        "prompt_additions": persona.prompt_additions
    }

    # Include resolved template if available
    if persona.template:
        result["template"] = {
            "id": persona.template.id,
            "name": persona.template.name,
            "description": persona.template.description,
            "epistemic_lens": persona.template.epistemic_lens
        }

    # Include merged personality
    personality = persona.get_personality()
    result["personality"] = {
        "openness": personality.openness,
        "conscientiousness": personality.conscientiousness,
        "extraversion": personality.extraversion,
        "agreeableness": personality.agreeableness,
        "neuroticism": personality.neuroticism
    }

    return jsonify(result)


# ============================================================================
# API: Agents (Legacy - uses personas internally)
# ============================================================================

@app.route('/api/agents')
def get_agents():
    """Get list of all agents with metadata.

    This endpoint is maintained for backward compatibility.
    Internally uses the persona system.
    """
    agents = []

    # Load personas
    loader = get_persona_loader()
    personas = loader.load_all()

    for persona_id, persona in personas.items():
        agents.append({
            "id": persona_id,
            "name": persona.name,
            "full_name": persona.name,
            "color": persona.color,
            "description": persona.description,
            "template_id": persona.template_id,
            "template_name": persona.template.name if persona.template else None,
            "is_human": False
        })

    # Add human
    agents.append({
        "id": "human",
        "name": "You",
        "full_name": "human-participant",
        "color": "#B49070",
        "description": "Human participant",
        "is_human": True
    })

    return jsonify({"agents": agents})


# ============================================================================
# API: Session management
# ============================================================================

@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new dialogue session.

    Request body:
        provocation: str - Opening question for the circle
        personas: list[str] - Optional list of persona IDs to include (2-7)
        include_human: bool - Whether to include human participant (default: True)
        seed: int - Random seed (default: 42)
        config: str - Config name for model assignments (default: 'multi_model')
    """
    data = request.get_json() or {}
    provocation = data.get('provocation', 'What does it mean to live well?')
    seed = data.get('seed', 42)
    config_name = data.get('config', 'multi_model')
    persona_ids = data.get('personas')  # Optional: list of persona IDs
    include_human = data.get('include_human', True)

    # Validate persona selection if provided
    if persona_ids:
        if not isinstance(persona_ids, list):
            return jsonify({"error": "personas must be a list of persona IDs"}), 400
        if len(persona_ids) < 2:
            return jsonify({"error": "At least 2 personas required"}), 400
        if len(persona_ids) > 7:
            return jsonify({"error": "Maximum 7 personas allowed"}), 400

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
        agents_dir=AGENTS_DIR,
        persona_ids=persona_ids,
        include_human=include_human
    )

    # Store session
    sessions[session.session_id] = session

    return jsonify({
        "session_id": session.session_id,
        "provocation": provocation,
        "personas": persona_ids or list(session.agents.keys()),
        "include_human": include_human,
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
            "color": get_persona_color(agent_id)
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
    """Format a TurnEvent, StateEvent, or MetricsEvent as SSE data."""
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
            "color": get_persona_color(event.agent_id)
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

    elif isinstance(event, MetricsEvent):
        data = {
            "type": "metrics",
            "turn_number": event.turn_number,
            "basin": event.basin,
            "basin_confidence": event.basin_confidence,
            "integrity_score": event.integrity_score,
            "integrity_label": event.integrity_label,
            "psi_semantic": event.psi_semantic,
            "psi_temporal": event.psi_temporal,
            "psi_affective": event.psi_affective,
            "voice_distinctiveness": event.voice_distinctiveness,
            "velocity_magnitude": event.velocity_magnitude
        }
        return f"event: metrics\ndata: {json.dumps(data)}\n\n"

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


@app.route('/api/session/<session_id>/inject', methods=['POST'])
def inject_prompt(session_id: str):
    """Inject a researcher prompt into the dialogue context.

    This adds a prompt that agents will see but does NOT count as a turn.
    Useful for researcher interventions like "Challenge this" or "Ask Luma".
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json() or {}
    content = data.get('content', '').strip()

    if not content:
        return jsonify({"error": "content is required"}), 400

    session = sessions[session_id]
    session.inject_prompt(content)

    return jsonify({
        "status": "injected",
        "content": content,
        "turn": session.turn_number
    })


@app.route('/api/session/<session_id>/continue', methods=['POST'])
def continue_session(session_id: str):
    """Continue the dialogue without human input.

    Used when it's the human's turn but they want to skip and let
    the AI agents continue the conversation.
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    session._trigger_next_response()

    return jsonify({"status": "continued"})


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
