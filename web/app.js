/**
 * MASE Circle - Interactive Dialogue Client
 *
 * Handles SSE streaming, state management, and UI updates
 * for human participation in multi-agent dialogues.
 */

// ============================================================================
// Configuration
// ============================================================================

const API_BASE = '';  // Same origin

const AGENT_COLORS = {
    luma: '#F59E0B',
    elowen: '#10B981',
    orin: '#3B82F6',
    nyra: '#8B5CF6',
    ilya: '#06B6D4',
    sefi: '#F97316',
    tala: '#EF4444',
    human: '#E5E7EB'
};

const AGENT_DESCRIPTIONS = {
    luma: 'Child voice, moral clarity',
    elowen: 'Ecological wisdom, kincentric',
    orin: 'Systems thinking, cybernetics',
    nyra: 'Moral imagination, design fiction',
    ilya: 'Liminal guide, posthuman',
    sefi: 'Policy pragmatist, governance',
    tala: 'Capitalist realist, markets',
    human: 'Human participant'
};


// ============================================================================
// State
// ============================================================================

let state = {
    sessionId: null,
    agents: [],
    history: [],
    currentState: 'idle',
    nextSpeaker: null,
    eventSource: null,
    turnCounts: {}
};


// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    // Status
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.querySelector('.status-text'),

    // Screens
    startScreen: document.getElementById('start-screen'),
    dialogueView: document.getElementById('dialogue-view'),

    // Start
    provocationInput: document.getElementById('provocation-input'),
    startBtn: document.getElementById('start-btn'),

    // Agents
    agentsList: document.getElementById('agents-list'),

    // Controls
    newSessionBtn: document.getElementById('new-session-btn'),
    pauseBtn: document.getElementById('pause-btn'),
    resumeBtn: document.getElementById('resume-btn'),

    // Session info
    sessionIdEl: document.getElementById('session-id'),
    turnCountEl: document.getElementById('turn-count'),

    // Dialogue
    provocationBanner: document.getElementById('provocation-banner'),
    provocationText: document.getElementById('provocation-text'),
    dialogueMessages: document.getElementById('dialogue-messages'),
    speakingIndicator: document.getElementById('speaking-indicator'),

    // Input
    inputArea: document.getElementById('input-area'),
    humanInput: document.getElementById('human-input'),
    sendBtn: document.getElementById('send-btn'),
    quickPrompts: document.getElementById('quick-prompts')
};


// ============================================================================
// API Functions
// ============================================================================

async function fetchAgents() {
    try {
        const response = await fetch(`${API_BASE}/api/agents`);
        const data = await response.json();
        state.agents = data.agents;
        renderAgents();
    } catch (error) {
        console.error('Failed to fetch agents:', error);
    }
}

async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();

        if (data.ollama_running) {
            setStatus('connected', 'Connected');
        } else {
            setStatus('disconnected', 'Ollama offline');
        }

        return data;
    } catch (error) {
        setStatus('disconnected', 'Server offline');
        return null;
    }
}

async function startSession(provocation) {
    try {
        const response = await fetch(`${API_BASE}/api/session/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provocation })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to start session');
        }

        const data = await response.json();
        state.sessionId = data.session_id;
        state.history = [];
        state.turnCounts = {};

        // Update UI
        elements.sessionIdEl.textContent = state.sessionId.slice(0, 8) + '...';
        elements.provocationText.textContent = provocation;

        // Switch to dialogue view
        showDialogueView();

        // Start streaming
        connectSSE();

        return data;
    } catch (error) {
        console.error('Failed to start session:', error);
        alert('Failed to start session: ' + error.message);
    }
}

async function submitHumanTurn(content) {
    if (!state.sessionId) return;

    try {
        const response = await fetch(`${API_BASE}/api/session/${state.sessionId}/human`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        });

        if (!response.ok) {
            throw new Error('Failed to submit turn');
        }

        // Clear input - the message will arrive via SSE from the queue
        elements.humanInput.value = '';

        // No need to reconnect - SSE connection stays open and will
        // receive the human turn event from the server's queue

    } catch (error) {
        console.error('Failed to submit turn:', error);
    }
}

async function invokeAgent(agentId) {
    if (!state.sessionId) return;

    try {
        await fetch(`${API_BASE}/api/session/${state.sessionId}/invoke`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_id: agentId })
        });

        // No need to reconnect - SSE connection stays open
    } catch (error) {
        console.error('Failed to invoke agent:', error);
    }
}

async function pauseSession() {
    if (!state.sessionId) return;

    try {
        await fetch(`${API_BASE}/api/session/${state.sessionId}/pause`, {
            method: 'POST'
        });
        state.currentState = 'paused';
        updateControls();
    } catch (error) {
        console.error('Failed to pause session:', error);
    }
}

async function resumeSession() {
    if (!state.sessionId) return;

    try {
        await fetch(`${API_BASE}/api/session/${state.sessionId}/resume`, {
            method: 'POST'
        });
        state.currentState = 'running';
        updateControls();
        // No need to reconnect - SSE connection stays open
    } catch (error) {
        console.error('Failed to resume session:', error);
    }
}

async function endSession() {
    if (!state.sessionId) return;

    try {
        await fetch(`${API_BASE}/api/session/${state.sessionId}/end`, {
            method: 'POST'
        });
    } catch (error) {
        console.error('Failed to end session:', error);
    }

    // Disconnect SSE
    if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
    }

    // Reset state
    state.sessionId = null;
    state.history = [];
    state.currentState = 'idle';
    state.turnCounts = {};

    // Show start screen
    showStartScreen();
}


// ============================================================================
// SSE Streaming
// ============================================================================

function connectSSE() {
    // Close existing connection
    if (state.eventSource) {
        state.eventSource.close();
    }

    if (!state.sessionId) return;

    const url = `${API_BASE}/api/session/${state.sessionId}/stream`;
    state.eventSource = new EventSource(url);

    state.eventSource.addEventListener('turn', (event) => {
        const data = JSON.parse(event.data);
        handleTurnEvent(data);
    });

    state.eventSource.addEventListener('state', (event) => {
        const data = JSON.parse(event.data);
        handleStateEvent(data);
    });

    state.eventSource.addEventListener('error', (event) => {
        console.error('SSE error:', event);
        hideSpeakingIndicator();

        // Auto-reconnect after a delay if session still active
        if (state.sessionId && state.currentState !== 'complete') {
            setTimeout(() => {
                if (state.sessionId && (!state.eventSource || state.eventSource.readyState === EventSource.CLOSED)) {
                    console.log('Auto-reconnecting SSE...');
                    connectSSE();
                }
            }, 2000);
        }
    });
}


// ============================================================================
// Event Handlers
// ============================================================================

function handleTurnEvent(data) {
    // Hide speaking indicator
    hideSpeakingIndicator();

    // Add message
    addMessage(data);

    // Update turn count
    state.turnCounts[data.agent_id] = (state.turnCounts[data.agent_id] || 0) + 1;
    elements.turnCountEl.textContent = state.history.length;

    // Update agents display
    renderAgents();
}

function handleStateEvent(data) {
    state.currentState = data.state;
    state.nextSpeaker = data.next_speaker;

    setStatus(data.state, data.message || data.state);
    updateControls();
    renderAgents();  // Update sidebar to show next speaker

    if (data.state === 'running' && data.next_speaker && data.next_speaker !== 'human') {
        // Show who is speaking
        showSpeakingIndicator(data.next_speaker);
    } else if (data.state === 'awaiting_human') {
        // Human's turn
        hideSpeakingIndicator();
        focusInput();
    } else {
        hideSpeakingIndicator();
    }
}


// ============================================================================
// UI Rendering
// ============================================================================

function renderAgents() {
    elements.agentsList.innerHTML = '';

    const agents = state.agents.length > 0 ? state.agents : [
        { id: 'elowen', name: 'Elowen' },
        { id: 'orin', name: 'Orin' },
        { id: 'nyra', name: 'Nyra' },
        { id: 'ilya', name: 'Ilya' },
        { id: 'sefi', name: 'Sefi' },
        { id: 'tala', name: 'Tala' },
        { id: 'luma', name: 'Luma' },
        { id: 'human', name: 'You', is_human: true }
    ];

    for (const agent of agents) {
        const item = document.createElement('div');
        item.className = 'agent-item';
        if (state.nextSpeaker === agent.id) {
            item.classList.add('speaking');
        }

        const color = agent.color || AGENT_COLORS[agent.id] || '#888888';
        const desc = agent.description || AGENT_DESCRIPTIONS[agent.id] || '';
        const turns = state.turnCounts[agent.id] || 0;

        item.innerHTML = `
            <div class="agent-dot" style="background: ${color}"></div>
            <div class="agent-info">
                <div class="agent-name">${agent.name}</div>
                <div class="agent-desc">${desc}</div>
            </div>
            ${turns > 0 ? `<div class="agent-turns">${turns}</div>` : ''}
        `;

        // Click to invoke (except human)
        if (!agent.is_human && state.sessionId) {
            item.style.cursor = 'pointer';
            item.addEventListener('click', () => invokeAgent(agent.id));
        }

        elements.agentsList.appendChild(item);
    }
}

function addMessage(data) {
    state.history.push(data);

    const message = document.createElement('div');
    message.className = 'message';
    if (data.is_human) {
        message.classList.add('human');
    }

    const color = data.color || AGENT_COLORS[data.agent_id] || '#888888';
    const initial = (data.agent_name || data.agent_id)[0].toUpperCase();

    // Format content with paragraphs
    const contentHtml = formatContent(data.content);

    message.innerHTML = `
        <div class="message-avatar" style="background: ${color}">${initial}</div>
        <div class="message-body">
            <div class="message-header">
                <span class="message-name" style="color: ${color}">${data.agent_name || data.agent_id}</span>
                ${data.latency_ms ? `<span class="message-meta">${Math.round(data.latency_ms)}ms</span>` : ''}
            </div>
            <div class="message-content">${contentHtml}</div>
        </div>
    `;

    elements.dialogueMessages.appendChild(message);

    // Scroll to bottom
    elements.dialogueMessages.scrollTop = elements.dialogueMessages.scrollHeight;
}

function formatContent(text) {
    if (!text) return '';

    // Split into paragraphs
    const paragraphs = text.split(/\n\n+/);

    return paragraphs
        .map(p => {
            // Basic markdown: bold and italic
            let html = p
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                .replace(/\*([^*]+)\*/g, '<em>$1</em>')
                .replace(/\n/g, '<br>');
            return `<p>${html}</p>`;
        })
        .join('');
}

function showSpeakingIndicator(agentId) {
    const agent = state.agents.find(a => a.id === agentId) || { name: agentId };
    const color = AGENT_COLORS[agentId] || '#888888';

    elements.speakingIndicator.classList.remove('hidden');
    elements.speakingIndicator.querySelector('.speaking-dot').style.background = color;
    elements.speakingIndicator.querySelector('.speaking-text').textContent =
        `${agent.name || agentId} is speaking...`;
}

function hideSpeakingIndicator() {
    elements.speakingIndicator.classList.add('hidden');
}

function setStatus(statusClass, text) {
    elements.statusIndicator.className = 'status-indicator ' + statusClass;
    elements.statusText.textContent = text;
}

function updateControls() {
    const hasSession = !!state.sessionId;
    const isRunning = state.currentState === 'running';
    const isPaused = state.currentState === 'paused';
    const isAwaiting = state.currentState === 'awaiting_human';

    elements.pauseBtn.disabled = !hasSession || !isRunning;
    elements.resumeBtn.disabled = !hasSession || !isPaused;
    elements.sendBtn.disabled = !hasSession;

    // Enable input when awaiting human
    if (isAwaiting) {
        elements.humanInput.placeholder = "It's your turn to speak...";
    } else {
        elements.humanInput.placeholder = "Add your voice to the circle...";
    }
}

function focusInput() {
    elements.humanInput.focus();
}

function showStartScreen() {
    elements.startScreen.classList.remove('hidden');
    elements.dialogueView.classList.add('hidden');
    updateControls();
}

function showDialogueView() {
    elements.startScreen.classList.add('hidden');
    elements.dialogueView.classList.remove('hidden');
    elements.dialogueMessages.innerHTML = '';
    updateControls();
}


// ============================================================================
// Event Listeners
// ============================================================================

// Start button
elements.startBtn.addEventListener('click', () => {
    const provocation = elements.provocationInput.value.trim();
    if (provocation) {
        startSession(provocation);
    }
});

// Enter key in provocation input
elements.provocationInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        elements.startBtn.click();
    }
});

// Example provocations
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        elements.provocationInput.value = btn.textContent;
        elements.provocationInput.focus();
    });
});

// New session button
elements.newSessionBtn.addEventListener('click', () => {
    if (state.sessionId) {
        if (confirm('End current session and start a new one?')) {
            endSession();
        }
    } else {
        showStartScreen();
    }
});

// Pause/Resume buttons
elements.pauseBtn.addEventListener('click', pauseSession);
elements.resumeBtn.addEventListener('click', resumeSession);

// Send button
elements.sendBtn.addEventListener('click', () => {
    const content = elements.humanInput.value.trim();
    if (content) {
        submitHumanTurn(content);
    }
});

// Enter key in human input
elements.humanInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        elements.sendBtn.click();
    }
});

// Quick prompts
elements.quickPrompts.addEventListener('click', (e) => {
    const btn = e.target.closest('.quick-btn');
    if (!btn || !state.sessionId) return;

    const action = btn.dataset.action;

    switch (action) {
        case 'continue':
            // Reconnect if connection was lost
            if (!state.eventSource || state.eventSource.readyState === EventSource.CLOSED) {
                connectSSE();
            }
            break;

        case 'invoke':
            const agentId = btn.dataset.agent;
            if (agentId) {
                invokeAgent(agentId);
            }
            break;

        case 'challenge':
            elements.humanInput.value = "I want to push back on what was just said. ";
            focusInput();
            break;

        case 'common-ground':
            elements.humanInput.value = "I'm noticing some common threads here. ";
            focusInput();
            break;
    }
});


// ============================================================================
// Initialization
// ============================================================================

async function init() {
    await checkStatus();
    await fetchAgents();
    renderAgents();

    // Check status periodically
    setInterval(checkStatus, 30000);
}

// Start the app
init();
