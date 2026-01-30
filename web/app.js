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

// Fallback colors (used if API doesn't provide colors)
const FALLBACK_COLORS = {
    human: 'rgb(180, 144, 112)',    // warm tan - human voice
    researcher: 'rgb(160, 160, 180)' // cool grey - researcher interjection
};

// Colors loaded from personas API (populated on init)
let PERSONA_COLORS = {};
let PERSONA_DESCRIPTIONS = {};

// Perturbation prompt templates
const PERTURB_TEMPLATES = {
    challenge: "I notice you're all agreeing. What's the strongest counterargument?",
    luma: "Luma, what do you think about this?",
    shake: "Let's step back. What are we missing? What assumptions haven't we questioned?",
    consolidate: "Can someone summarize where we've landed and what remains unresolved?"
};

/**
 * Get color for a persona/agent ID.
 * Uses colors loaded from API, falls back to defaults.
 */
function getPersonaColor(id) {
    if (PERSONA_COLORS[id]) return PERSONA_COLORS[id];
    if (FALLBACK_COLORS[id]) return FALLBACK_COLORS[id];
    return '#888888';
}

/**
 * Get description for a persona/agent ID.
 */
function getPersonaDescription(id) {
    return PERSONA_DESCRIPTIONS[id] || '';
}


// ============================================================================
// State
// ============================================================================

let state = {
    sessionId: null,
    agents: [],
    personas: [],           // All available personas
    selectedPersonas: [],   // IDs of selected personas for next session
    includeHuman: true,     // Whether to include human in circle
    history: [],
    currentState: 'idle',
    nextSpeaker: null,
    eventSource: null,
    turnCounts: {},
    currentView: 'circle',
    lastBasin: null  // For basin transition detection
};


// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    // Status
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.querySelector('.status-text'),

    // Navigation
    navTabs: document.querySelectorAll('.nav-tab'),
    circleView: document.getElementById('circle-view'),
    sessionsView: document.getElementById('sessions-view'),

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
    endBtn: document.getElementById('end-btn'),

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
    quickPrompts: document.getElementById('quick-prompts'),

    // Analysis
    analysisView: document.getElementById('analysis-view'),
    analysisDialogueTab: document.getElementById('analysis-dialogue-tab'),
    analysisMetricsTab: document.getElementById('analysis-metrics-tab'),
    analysisDialogueMessages: document.getElementById('analysis-dialogue-messages'),
    analysisSummary: document.getElementById('analysis-summary'),
    basinChart: document.getElementById('basin-chart'),
    metricsGrid: document.getElementById('metrics-grid'),
    agentStats: document.getElementById('agent-stats'),
    closeAnalysisBtn: document.getElementById('close-analysis-btn'),

    // Modal & Loading
    endModal: document.getElementById('end-modal'),
    cancelEndBtn: document.getElementById('cancel-end-btn'),
    confirmEndBtn: document.getElementById('confirm-end-btn'),
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    loadingDetail: document.getElementById('loading-detail'),
    toastContainer: document.getElementById('toast-container'),

    // Live Metrics Panel
    metricsPanel: document.getElementById('metrics-panel'),
    liveBasin: document.getElementById('live-basin'),
    liveIntegrity: document.getElementById('live-integrity'),
    liveVoiceDist: document.getElementById('live-voice-dist'),
    psiSemanticFill: document.getElementById('psi-semantic-fill'),
    psiTemporalFill: document.getElementById('psi-temporal-fill'),
    psiAffectiveFill: document.getElementById('psi-affective-fill'),

    // Perturbation Panel
    perturbationPanel: document.getElementById('perturbation-panel'),
    customPrompt: document.getElementById('custom-prompt'),
    injectCustomBtn: document.getElementById('inject-custom-btn')
};


// ============================================================================
// API Functions
// ============================================================================

async function fetchPersonas() {
    try {
        const response = await fetch(`${API_BASE}/api/personas`);
        const data = await response.json();
        state.personas = data.personas;

        // Populate color and description maps from API data
        PERSONA_COLORS = {};
        PERSONA_DESCRIPTIONS = {};
        for (const persona of data.personas) {
            PERSONA_COLORS[persona.id] = persona.color;
            PERSONA_DESCRIPTIONS[persona.id] = persona.description;
        }

        // Default to all personas selected
        if (state.selectedPersonas.length === 0) {
            state.selectedPersonas = data.personas.map(p => p.id);
        }

        renderPersonaSelection();
    } catch (error) {
        console.error('Failed to fetch personas:', error);
    }
}

async function fetchAgents() {
    try {
        const response = await fetch(`${API_BASE}/api/agents`);
        const data = await response.json();
        state.agents = data.agents;

        // Also populate color maps from agents data
        for (const agent of data.agents) {
            if (agent.color) {
                PERSONA_COLORS[agent.id] = agent.color;
            }
            if (agent.description) {
                PERSONA_DESCRIPTIONS[agent.id] = agent.description;
            }
        }

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
    // Validate persona selection
    if (state.selectedPersonas.length < 2) {
        showToast('Selection Required', 'Select at least 2 personas for the circle', null, null);
        return;
    }
    if (state.selectedPersonas.length > 7) {
        showToast('Too Many Personas', 'Maximum 7 personas allowed', null, null);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/session/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provocation,
                personas: state.selectedPersonas,
                include_human: state.includeHuman
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to start session');
        }

        const data = await response.json();
        state.sessionId = data.session_id;
        state.history = [];
        state.turnCounts = {};

        // Update agent list from session response
        state.agents = data.agents;
        for (const agent of data.agents) {
            if (agent.color) {
                PERSONA_COLORS[agent.id] = agent.color;
            }
            if (agent.description) {
                PERSONA_DESCRIPTIONS[agent.id] = agent.description;
            }
        }

        // Update UI
        elements.sessionIdEl.textContent = state.sessionId.slice(0, 8) + '...';
        elements.provocationText.textContent = provocation;

        // Switch to dialogue view
        showDialogueView();

        // Render agents sidebar with selected personas
        renderAgents();

        // Start streaming
        connectSSE();

        return data;
    } catch (error) {
        console.error('Failed to start session:', error);
        showToast('Error', 'Failed to start session: ' + error.message, null, null);
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

        // Ensure SSE connection is alive to receive the queued turn event
        if (!state.eventSource || state.eventSource.readyState === EventSource.CLOSED) {
            console.log('SSE connection closed, reconnecting...');
            connectSSE();
        }

    } catch (error) {
        console.error('Failed to submit turn:', error);
        showToast('Error', 'Failed to send message', null, null);
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

async function continueDialogue() {
    if (!state.sessionId) return;

    try {
        await fetch(`${API_BASE}/api/session/${state.sessionId}/continue`, {
            method: 'POST'
        });

        // Ensure SSE connection is alive
        if (!state.eventSource || state.eventSource.readyState === EventSource.CLOSED) {
            connectSSE();
        }
    } catch (error) {
        console.error('Failed to continue dialogue:', error);
    }
}

async function injectPrompt(template) {
    if (!state.sessionId) return;

    const content = PERTURB_TEMPLATES[template] || template;

    try {
        await fetch(`${API_BASE}/api/session/${state.sessionId}/inject`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        });

        // If template is 'luma', invoke Luma specifically to respond
        if (template === 'luma') {
            await invokeAgent('luma');
        }
        // Server-side inject_prompt now triggers response automatically

        // Show feedback
        showToast('Prompt Injected', content.slice(0, 50) + '...', null, null);

    } catch (error) {
        console.error('Failed to inject prompt:', error);
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

// Store pending analysis for toast click
let pendingAnalysis = null;

function showEndModal() {
    if (!state.sessionId) return;
    elements.endModal.classList.remove('hidden');
}

function hideEndModal() {
    elements.endModal.classList.add('hidden');
}

function showLoading(text, detail) {
    elements.loadingText.textContent = text;
    elements.loadingDetail.textContent = detail;
    elements.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    elements.loadingOverlay.classList.add('hidden');
}

function showToast(title, message, action, onClick) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <div class="toast-icon">✓</div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
            ${action ? `<div class="toast-action">${action}</div>` : ''}
        </div>
    `;

    toast.addEventListener('click', () => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
        if (onClick) onClick();
    });

    elements.toastContainer.appendChild(toast);

    // Auto-dismiss after 10 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.classList.add('toast-exit');
            setTimeout(() => toast.remove(), 300);
        }
    }, 10000);
}

async function endSession() {
    if (!state.sessionId) return;

    // Show modal for confirmation
    showEndModal();
}

async function confirmEndSession() {
    if (!state.sessionId) return;

    // Capture session ID before it gets cleared
    const endedSessionId = state.sessionId;

    hideEndModal();
    showLoading('Ending session...', 'Saving dialogue to disk');

    let analysisData = null;

    try {
        // Update loading state
        setTimeout(() => {
            if (!elements.loadingOverlay.classList.contains('hidden')) {
                elements.loadingText.textContent = 'Analyzing session...';
                elements.loadingDetail.textContent = 'Computing embeddings (this may take a moment)';
            }
        }, 500);

        setTimeout(() => {
            if (!elements.loadingOverlay.classList.contains('hidden')) {
                elements.loadingDetail.textContent = 'Detecting basins and coherence patterns';
            }
        }, 2000);

        const response = await fetch(`${API_BASE}/api/session/${state.sessionId}/end`, {
            method: 'POST'
        });
        const data = await response.json();
        analysisData = data.analysis;
    } catch (error) {
        console.error('Failed to end session:', error);
        hideLoading();
        showToast('Error', 'Failed to end session', null, null);
        return;
    }

    // Disconnect SSE
    if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
    }

    // Reset state
    state.sessionId = null;
    state.currentState = 'idle';

    hideLoading();

    // Reset state
    state.history = [];
    state.turnCounts = {};

    // Show analysis if available
    if (analysisData && !analysisData.error) {
        // Show toast notification
        const dominantBasin = analysisData.dominant_basin || 'Unknown';
        const nTurns = analysisData.n_turns || 0;
        showToast(
            'Analysis Complete',
            `${nTurns} turns analyzed. Dominant basin: ${dominantBasin}`,
            'Click to view in Sessions →',
            () => {
                // Switch to Sessions view and select this session
                switchView('sessions');
                setTimeout(() => {
                    SessionsBrowser.loadSessions().then(() => {
                        SessionsBrowser.selectSession(endedSessionId);
                    });
                }, 100);
            }
        );

        showStartScreen();
    } else {
        showStartScreen();

        if (analysisData?.error) {
            showToast('Analysis Failed', analysisData.error, null, null);
        }
    }
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

    state.eventSource.addEventListener('metrics', (event) => {
        const data = JSON.parse(event.data);
        handleMetricsEvent(data);
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

function handleMetricsEvent(data) {
    // Update basin display
    if (elements.liveBasin) {
        // Shorten basin name for display
        const basinShort = data.basin
            .replace('Collaborative ', '')
            .replace('Cognitive ', '')
            .replace('Sycophantic ', '')
            .replace('Generative ', '')
            .replace('Reflexive ', '')
            .replace('Creative ', '');
        elements.liveBasin.textContent = basinShort;
        elements.liveBasin.className = 'basin-badge basin-' + data.basin.toLowerCase().replace(/ /g, '-');
    }

    // Update integrity display
    if (elements.liveIntegrity) {
        elements.liveIntegrity.textContent = data.integrity_label;
        elements.liveIntegrity.className = 'integrity-indicator integrity-' + data.integrity_label;
    }

    // Update voice distinctiveness
    if (elements.liveVoiceDist) {
        elements.liveVoiceDist.textContent = data.voice_distinctiveness.toFixed(2);
    }

    // Update Psi bars
    updatePsiBar(elements.psiSemanticFill, data.psi_semantic);
    updatePsiBar(elements.psiTemporalFill, data.psi_temporal);
    updatePsiBar(elements.psiAffectiveFill, data.psi_affective);

    // Toast on basin transition
    if (state.lastBasin && state.lastBasin !== data.basin) {
        showToast('Basin Shift', `${state.lastBasin} → ${data.basin}`, null, null);
    }
    state.lastBasin = data.basin;
}

function updatePsiBar(element, value) {
    if (!element) return;

    // Value should be in range [-1, 1], normalize to [0, 100]
    // We display centered at 50%, so:
    // - value = 0 → width 50%, positioned left 0% (neutral)
    // - value = 1 → width 50%, positioned left 50% (full positive)
    // - value = -1 → width 50%, positioned left 0%, but we show on left side

    // Actually, let's use a simpler approach: position the fill based on value
    // The fill always starts from center and extends in the direction of the value
    const normalized = Math.max(-1, Math.min(1, value || 0));
    const percentage = Math.abs(normalized) * 50;

    if (normalized >= 0) {
        element.style.left = '50%';
        element.style.width = percentage + '%';
        element.style.background = 'var(--accent-success)';
    } else {
        element.style.left = (50 - percentage) + '%';
        element.style.width = percentage + '%';
        element.style.background = 'var(--accent-warning)';
    }
}


// ============================================================================
// UI Rendering
// ============================================================================

function renderPersonaSelection() {
    const container = document.getElementById('persona-selection');
    if (!container) return;

    container.innerHTML = '';

    for (const persona of state.personas) {
        const isSelected = state.selectedPersonas.includes(persona.id);
        const card = document.createElement('div');
        card.className = `persona-card ${isSelected ? 'selected' : ''}`;
        card.dataset.personaId = persona.id;

        const color = persona.color || getPersonaColor(persona.id);

        card.innerHTML = `
            <div class="persona-card-header">
                <div class="persona-dot" style="background: ${color}"></div>
                <div class="persona-name">${persona.name}</div>
            </div>
            <div class="persona-template">${persona.template_name || ''}</div>
            <div class="persona-desc">${persona.description || ''}</div>
        `;

        card.addEventListener('click', () => togglePersonaSelection(persona.id));
        container.appendChild(card);
    }

    updatePersonaSelectionCount();
}

function togglePersonaSelection(personaId) {
    const index = state.selectedPersonas.indexOf(personaId);
    if (index === -1) {
        // Add to selection
        if (state.selectedPersonas.length < 7) {
            state.selectedPersonas.push(personaId);
        } else {
            showToast('Limit Reached', 'Maximum 7 personas allowed', null, null);
            return;
        }
    } else {
        // Remove from selection
        if (state.selectedPersonas.length > 2) {
            state.selectedPersonas.splice(index, 1);
        } else {
            showToast('Minimum Required', 'At least 2 personas required', null, null);
            return;
        }
    }

    renderPersonaSelection();
}

function updatePersonaSelectionCount() {
    const countEl = document.getElementById('persona-count');
    const startBtn = document.getElementById('start-btn');

    if (countEl) {
        countEl.textContent = `${state.selectedPersonas.length} selected`;
    }

    if (startBtn) {
        const count = state.selectedPersonas.length;
        const humanText = state.includeHuman ? ' + you' : '';
        startBtn.innerHTML = `<i class="ph ph-play-circle"></i> Begin Circle (${count} personas${humanText})`;
        startBtn.disabled = count < 2;
    }
}

function renderAgents() {
    elements.agentsList.innerHTML = '';

    // Use agents from current session, or personas as fallback
    let agents = state.agents;
    if (!agents || agents.length === 0) {
        // Convert personas to agent-like structure
        agents = state.personas.map(p => ({
            id: p.id,
            name: p.name,
            color: p.color,
            description: p.description,
            is_human: false
        }));
        // Add human if included
        if (state.includeHuman) {
            agents.push({
                id: 'human',
                name: 'You',
                color: getPersonaColor('human'),
                description: 'Human participant',
                is_human: true
            });
        }
    }

    for (const agent of agents) {
        const item = document.createElement('div');
        item.className = 'agent-item';
        if (state.nextSpeaker === agent.id) {
            item.classList.add('speaking');
        }

        const color = agent.color || getPersonaColor(agent.id);
        const desc = agent.description || getPersonaDescription(agent.id);
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
    if (data.agent_id === 'researcher') {
        message.classList.add('researcher');
    }

    const color = data.color || getPersonaColor(data.agent_id);
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
    const color = getPersonaColor(agentId);

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
    elements.endBtn.disabled = !hasSession;
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
    elements.analysisView.classList.add('hidden');
    state.history = [];
    state.turnCounts = {};
    updateControls();
}

function showDialogueView() {
    elements.startScreen.classList.add('hidden');
    elements.dialogueView.classList.remove('hidden');
    elements.analysisView.classList.add('hidden');
    elements.dialogueMessages.innerHTML = '';
    updateControls();
}

function showAnalysisView(analysis) {
    elements.startScreen.classList.add('hidden');
    elements.dialogueView.classList.add('hidden');
    elements.analysisView.classList.remove('hidden');

    // Render dialogue from state.history (preserved before reset)
    renderAnalysisDialogue();

    // Render analysis summary
    renderAnalysisSummary(analysis);
    renderBasinChart(analysis);
    renderMetrics(analysis);
    renderAgentStats(analysis);

    // Default to dialogue tab
    switchAnalysisTab('dialogue');
}

function renderAnalysisDialogue() {
    if (!elements.analysisDialogueMessages) return;

    // Use pendingHistory which we preserve before clearing state.history
    const history = window.pendingHistory || [];

    elements.analysisDialogueMessages.innerHTML = history.map(turn => {
        const color = turn.color || getPersonaColor(turn.agent_id);
        const initial = (turn.agent_name || turn.agent_id || 'U')[0].toUpperCase();
        const contentHtml = formatContent(turn.content);

        return `
            <div class="message ${turn.is_human ? 'human' : ''}">
                <div class="message-avatar" style="background: ${color}">${initial}</div>
                <div class="message-body">
                    <div class="message-header">
                        <span class="message-name" style="color: ${color}">${turn.agent_name || turn.agent_id}</span>
                        ${turn.latency_ms ? `<span class="message-meta">${Math.round(turn.latency_ms)}ms</span>` : ''}
                    </div>
                    <div class="message-content">${contentHtml}</div>
                </div>
            </div>
        `;
    }).join('');
}

function switchAnalysisTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.analysis-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Show/hide content
    if (elements.analysisDialogueTab) {
        elements.analysisDialogueTab.classList.toggle('hidden', tabName !== 'dialogue');
    }
    if (elements.analysisMetricsTab) {
        elements.analysisMetricsTab.classList.toggle('hidden', tabName !== 'analysis');
    }
}

function renderAnalysisSummary(analysis) {
    const dominant = analysis.dominant_basin || 'Unknown';
    const percentage = ((analysis.dominant_basin_percentage || 0) * 100).toFixed(0);
    const pattern = getCoherencePattern(analysis);

    elements.analysisSummary.innerHTML = `
        <div class="summary-card">
            <div class="summary-stat">
                <span class="stat-value">${analysis.n_turns || 0}</span>
                <span class="stat-label">Turns</span>
            </div>
            <div class="summary-stat">
                <span class="stat-value">${dominant}</span>
                <span class="stat-label">Dominant Basin (${percentage}%)</span>
            </div>
            <div class="summary-stat">
                <span class="stat-value">${pattern}</span>
                <span class="stat-label">Coherence Pattern</span>
            </div>
            <div class="summary-stat">
                <span class="stat-value">${analysis.transition_count || 0}</span>
                <span class="stat-label">Basin Transitions</span>
            </div>
        </div>
    `;
}

function getCoherencePattern(analysis) {
    const patterns = analysis.coherence_pattern_distribution || {};
    const breathing = patterns.breathing || 0;
    const transitional = patterns.transitional || 0;
    const locked = patterns.locked || 0;

    if (breathing > transitional && breathing > locked) return 'Breathing';
    if (locked > transitional && locked > breathing) return 'Locked';
    return 'Transitional';
}

function renderBasinChart(analysis) {
    const sequence = analysis.basin_sequence || [];
    if (sequence.length === 0) {
        elements.basinChart.innerHTML = '<p>No basin data available</p>';
        return;
    }

    const basinColors = {
        'Deep Resonance': 'rgb(130, 155, 130)',       // sage moss
        'Collaborative Inquiry': 'rgb(100, 140, 160)', // slate blue
        'Cognitive Mimicry': 'rgb(195, 140, 95)',      // amber ochre
        'Reflexive Performance': 'rgb(205, 110, 70)',  // burnt orange
        'Sycophantic Convergence': 'rgb(175, 130, 140)', // dusty rose
        'Creative Dilation': 'rgb(160, 130, 180)',     // soft purple
        'Generative Conflict': 'rgb(195, 160, 95)',    // warm gold
        'Dissociation': 'rgb(120, 120, 130)',          // cool grey
        'Transitional': 'rgb(150, 150, 150)'           // neutral grey
    };

    const bars = sequence.map((basin, i) => {
        const color = basinColors[basin] || '#888888';
        return `<div class="basin-bar" style="background: ${color}" title="Turn ${i + 1}: ${basin}"></div>`;
    }).join('');

    elements.basinChart.innerHTML = `
        <div class="basin-timeline">${bars}</div>
        <div class="basin-legend">
            ${Object.entries(analysis.basin_distribution || {}).map(([basin, count]) =>
                `<span class="legend-item">
                    <span class="legend-dot" style="background: ${basinColors[basin] || '#888'}"></span>
                    ${basin}: ${count}
                </span>`
            ).join('')}
        </div>
    `;
}

function renderMetrics(analysis) {
    const metrics = [
        { label: 'DFA α', value: (analysis.dfa_alpha || 0).toFixed(2), desc: 'Long-range correlation (0.5=noise, 1.0=pink)' },
        { label: 'Semantic Curvature', value: (analysis.semantic_curvature || 0).toFixed(2), desc: 'Trajectory complexity' },
        { label: 'Entropy Shift', value: (analysis.entropy_shift || 0).toFixed(2), desc: 'Semantic reorganization' },
        { label: 'Voice Distinctiveness', value: (analysis.voice_distinctiveness || 0).toFixed(2), desc: 'Agent differentiation' },
        { label: 'Inquiry Ratio', value: (analysis.inquiry_vs_mimicry_ratio || 0).toFixed(2), desc: 'Inquiry vs mimicry balance' },
        { label: 'Semantic Velocity', value: (analysis.semantic_velocity_mean || 0).toFixed(2), desc: 'Average semantic motion' }
    ];

    elements.metricsGrid.innerHTML = metrics.map(m => `
        <div class="metric-card">
            <div class="metric-value">${m.value}</div>
            <div class="metric-label">${m.label}</div>
            <div class="metric-desc">${m.desc}</div>
        </div>
    `).join('');
}

function renderAgentStats(analysis) {
    const agents = analysis.agents || [];
    const turnStates = analysis.turn_states || [];

    // Count turns per agent
    const agentTurns = {};
    turnStates.forEach(t => {
        agentTurns[t.agent_id] = (agentTurns[t.agent_id] || 0) + 1;
    });

    elements.agentStats.innerHTML = agents.map(agent => {
        const turns = agentTurns[agent] || 0;
        const color = getPersonaColor(agent);
        return `
            <div class="agent-stat-row">
                <span class="agent-dot" style="background: ${color}"></span>
                <span class="agent-name">${agent}</span>
                <span class="agent-turns">${turns} turns</span>
            </div>
        `;
    }).join('');
}


// ============================================================================
// Navigation
// ============================================================================

function switchView(viewName) {
    state.currentView = viewName;

    // Update nav tabs
    elements.navTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === viewName);
    });

    // Show/hide views
    elements.circleView.classList.toggle('hidden', viewName !== 'circle');
    elements.sessionsView.classList.toggle('hidden', viewName !== 'sessions');

    // Load sessions when switching to sessions view
    if (viewName === 'sessions') {
        SessionsBrowser.loadSessions();
    }
}


// ============================================================================
// Event Listeners
// ============================================================================

// Navigation tabs
elements.navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        switchView(tab.dataset.view);
    });
});

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

// New session button (if present)
if (elements.newSessionBtn) {
    elements.newSessionBtn.addEventListener('click', () => {
        if (state.sessionId) {
            if (confirm('End current session and start a new one?')) {
                endSession();
            }
        } else {
            showStartScreen();
        }
    });
}

// Pause/Resume/End buttons
elements.pauseBtn.addEventListener('click', pauseSession);
elements.resumeBtn.addEventListener('click', resumeSession);
elements.endBtn.addEventListener('click', showEndModal);

// End session modal
elements.cancelEndBtn.addEventListener('click', hideEndModal);
elements.confirmEndBtn.addEventListener('click', confirmEndSession);

// Close modal on overlay click
elements.endModal.addEventListener('click', (e) => {
    if (e.target === elements.endModal) hideEndModal();
});

// Close analysis and start new session
elements.closeAnalysisBtn.addEventListener('click', showStartScreen);

// Analysis tab switching
document.querySelectorAll('.analysis-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        switchAnalysisTab(tab.dataset.tab);
    });
});

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
            // Signal server to continue without human input
            continueDialogue();
            break;

        case 'common-ground':
            elements.humanInput.value = "I'm noticing some common threads here. ";
            focusInput();
            break;
    }
});

// Perturbation buttons
if (elements.perturbationPanel) {
    elements.perturbationPanel.addEventListener('click', (e) => {
        // Handle collapsible toggle
        if (e.target.closest('.collapsible-header')) {
            elements.perturbationPanel.classList.toggle('collapsed');
            return;
        }

        // Handle perturb buttons
        const btn = e.target.closest('.perturb-btn');
        if (btn && state.sessionId) {
            const template = btn.dataset.template;
            injectPrompt(template);
        }
    });
}

// Custom prompt injection
if (elements.injectCustomBtn) {
    elements.injectCustomBtn.addEventListener('click', () => {
        const content = elements.customPrompt.value.trim();
        if (content && state.sessionId) {
            injectPrompt(content);
            elements.customPrompt.value = '';
        }
    });
}

// Enter key in custom prompt
if (elements.customPrompt) {
    elements.customPrompt.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            elements.injectCustomBtn.click();
        }
    });
}


// ============================================================================
// Initialization
// ============================================================================

async function init() {
    // Initialize modules
    SessionsBrowser.init();

    // Initialize circle view
    await checkStatus();
    await fetchPersonas();  // Load personas for selection
    await fetchAgents();    // Load agents for sidebar display
    renderAgents();

    // Set up include human checkbox
    const includeHumanCheckbox = document.getElementById('include-human');
    if (includeHumanCheckbox) {
        includeHumanCheckbox.checked = state.includeHuman;
        includeHumanCheckbox.addEventListener('change', (e) => {
            state.includeHuman = e.target.checked;
            updatePersonaSelectionCount();
        });
    }

    // Check status periodically
    setInterval(checkStatus, 30000);
}

// Start the app
init();
