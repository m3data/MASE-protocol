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
    luma: 'rgb(195, 160, 95)',      // warm gold - child clarity
    elowen: 'rgb(130, 155, 130)',   // sage moss - ecological
    orin: 'rgb(100, 140, 160)',     // slate blue - systems
    nyra: 'rgb(175, 130, 160)',     // dusty mauve - imagination
    ilya: 'rgb(115, 155, 155)',     // soft teal - liminal
    sefi: 'rgb(195, 140, 95)',      // amber ochre - governance
    tala: 'rgb(205, 110, 70)',      // burnt orange - markets
    human: 'rgb(180, 144, 112)'     // warm tan - human voice
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
    turnCounts: {},
    // Sessions browser state
    currentView: 'circle',
    sessionsList: [],
    selectedSessionId: null
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

    // Sessions Browser
    sessionsCount: document.getElementById('sessions-count'),
    sessionsList: document.getElementById('sessions-list'),
    sessionDetailEmpty: document.getElementById('session-detail-empty'),
    sessionDetailContent: document.getElementById('session-detail-content'),
    detailSessionId: document.getElementById('detail-session-id'),
    detailTimestamp: document.getElementById('detail-timestamp'),
    detailProvocation: document.getElementById('detail-provocation'),
    detailTabs: document.querySelectorAll('.detail-tab'),
    detailDialogueTab: document.getElementById('detail-dialogue-tab'),
    detailAnalysisTab: document.getElementById('detail-analysis-tab'),
    detailDialogueMessages: document.getElementById('detail-dialogue-messages'),
    detailAnalysisContent: document.getElementById('detail-analysis-content')
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

    // Show analysis if available
    if (analysisData && !analysisData.error) {
        pendingAnalysis = analysisData;

        // Show toast notification
        const dominantBasin = analysisData.dominant_basin || 'Unknown';
        const nTurns = analysisData.n_turns || 0;
        showToast(
            'Analysis Complete',
            `${nTurns} turns analyzed. Dominant basin: ${dominantBasin}`,
            'Click to view results →',
            () => showAnalysisView(pendingAnalysis)
        );

        // Return to start screen (user can click toast to see results)
        state.history = [];
        state.turnCounts = {};
        showStartScreen();
    } else {
        // Reset remaining state
        state.history = [];
        state.turnCounts = {};
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

    // Render analysis summary
    renderAnalysisSummary(analysis);
    renderBasinChart(analysis);
    renderMetrics(analysis);
    renderAgentStats(analysis);
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
        const color = AGENT_COLORS[agent] || '#888888';
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
// Sessions Browser Functions
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

    // Load sessions list when switching to sessions view
    if (viewName === 'sessions') {
        loadSessionsList();
    }
}

async function loadSessionsList() {
    try {
        const response = await fetch(`${API_BASE}/api/sessions`);
        const data = await response.json();
        state.sessionsList = data.sessions || [];
        renderSessionsList();
    } catch (error) {
        console.error('Failed to load sessions:', error);
        elements.sessionsList.innerHTML = '<div class="sessions-empty">Failed to load sessions</div>';
    }
}

function renderSessionsList() {
    const sessions = state.sessionsList;
    elements.sessionsCount.textContent = `${sessions.length} session${sessions.length !== 1 ? 's' : ''}`;

    if (sessions.length === 0) {
        elements.sessionsList.innerHTML = '<div class="sessions-empty">No sessions yet. Start a dialogue in the Circle tab.</div>';
        return;
    }

    elements.sessionsList.innerHTML = sessions.map(session => {
        const isSelected = state.selectedSessionId === session.session_id;
        const timestamp = session.timestamp ? formatTimestamp(session.timestamp) : '';
        return `
            <div class="session-item ${isSelected ? 'selected' : ''}" data-session-id="${session.session_id}">
                <div class="session-item-header">
                    <span class="session-item-id">${session.session_id.slice(-8)}</span>
                    <div class="session-item-meta">
                        <span class="session-item-turns">${session.n_turns} turns</span>
                        <span class="session-item-status ${session.has_analysis ? 'has-analysis' : 'no-analysis'}">
                            ${session.has_analysis ? 'analyzed' : 'pending'}
                        </span>
                    </div>
                </div>
                <div class="session-item-provocation">${escapeHtml(session.provocation || 'No provocation')}</div>
                ${timestamp ? `<div class="session-item-timestamp">${timestamp}</div>` : ''}
            </div>
        `;
    }).join('');
}

function formatTimestamp(isoString) {
    try {
        const date = new Date(isoString);
        return date.toLocaleDateString(undefined, {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return '';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function selectSession(sessionId) {
    state.selectedSessionId = sessionId;

    // Update list selection
    document.querySelectorAll('.session-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.sessionId === sessionId);
    });

    // Show detail panel
    elements.sessionDetailEmpty.classList.add('hidden');
    elements.sessionDetailContent.classList.remove('hidden');

    // Load session data
    await loadSessionDetail(sessionId);
}

async function loadSessionDetail(sessionId) {
    try {
        // Load dialogue
        const dialogueResponse = await fetch(`${API_BASE}/api/sessions/${sessionId}/dialogue`);
        const dialogueData = await dialogueResponse.json();

        // Update header
        elements.detailSessionId.textContent = sessionId.slice(-8);
        elements.detailTimestamp.textContent = dialogueData.start_time ? formatTimestamp(dialogueData.start_time) : '';
        elements.detailProvocation.textContent = dialogueData.provocation || '';

        // Render dialogue
        renderDetailDialogue(dialogueData.turns || []);

        // Try to load analysis
        try {
            const analysisResponse = await fetch(`${API_BASE}/api/sessions/${sessionId}/analysis`);
            if (analysisResponse.ok) {
                const analysisData = await analysisResponse.json();
                renderDetailAnalysis(analysisData);
            } else {
                elements.detailAnalysisContent.innerHTML = '<p class="sessions-empty">No analysis available for this session.</p>';
            }
        } catch {
            elements.detailAnalysisContent.innerHTML = '<p class="sessions-empty">Failed to load analysis.</p>';
        }

    } catch (error) {
        console.error('Failed to load session detail:', error);
        elements.detailDialogueMessages.innerHTML = '<p class="sessions-empty">Failed to load session.</p>';
    }
}

function renderDetailDialogue(turns) {
    if (turns.length === 0) {
        elements.detailDialogueMessages.innerHTML = '<p class="sessions-empty">No dialogue content.</p>';
        return;
    }

    elements.detailDialogueMessages.innerHTML = turns.map(turn => {
        const agentId = turn.agent_id || 'unknown';
        const color = AGENT_COLORS[agentId] || '#888888';
        const initials = agentId.slice(0, 2).toUpperCase();
        const content = turn.content || '';

        return `
            <div class="message ${agentId === 'human' ? 'human' : ''}">
                <div class="message-avatar" style="background: ${color}">${initials}</div>
                <div class="message-body">
                    <div class="message-header">
                        <span class="message-name" style="color: ${color}">${agentId}</span>
                    </div>
                    <div class="message-content">${escapeHtml(content)}</div>
                </div>
            </div>
        `;
    }).join('');
}

function renderDetailAnalysis(analysis) {
    const metrics = analysis.semantic_metrics || {};
    const basinSequence = analysis.basin_sequence || [];
    const agents = analysis.agents || [];
    const turnStates = analysis.turn_states || [];

    // Build analysis HTML
    let html = '';

    // Summary
    html += `
        <div class="analysis-section">
            <h3>Summary</h3>
            <div class="summary-card">
                <div class="summary-stat">
                    <span class="stat-value">${turnStates.length}</span>
                    <span class="stat-label">Turns</span>
                </div>
                <div class="summary-stat">
                    <span class="stat-value">${agents.length}</span>
                    <span class="stat-label">Agents</span>
                </div>
                <div class="summary-stat">
                    <span class="stat-value">${(metrics.alpha || 0).toFixed(2)}</span>
                    <span class="stat-label">DFA α</span>
                </div>
            </div>
        </div>
    `;

    // Basin trajectory
    if (basinSequence.length > 0) {
        const basinColors = {
            'Deep Resonance': 'rgb(130, 155, 130)',
            'Collaborative Inquiry': 'rgb(100, 140, 160)',
            'Cognitive Mimicry': 'rgb(195, 140, 95)',
            'Reflexive Performance': 'rgb(205, 110, 70)',
            'Sycophantic Convergence': 'rgb(175, 130, 140)',
            'Creative Dilation': 'rgb(160, 130, 180)',
            'Generative Conflict': 'rgb(195, 160, 95)',
            'Dissociation': 'rgb(120, 120, 130)',
            'Transitional': 'rgb(150, 150, 150)'
        };

        const bars = basinSequence.map((basin, i) => {
            const color = basinColors[basin] || '#888888';
            return `<div class="basin-bar" style="background: ${color}" title="Turn ${i + 1}: ${basin}"></div>`;
        }).join('');

        html += `
            <div class="analysis-section">
                <h3>Basin Trajectory</h3>
                <div class="basin-timeline">${bars}</div>
            </div>
        `;
    }

    // Metrics
    html += `
        <div class="analysis-section">
            <h3>Semantic Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <span class="metric-value">${(metrics.curvature || 0).toFixed(3)}</span>
                    <span class="metric-label">Δκ Curvature</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">${(metrics.alpha || 0).toFixed(3)}</span>
                    <span class="metric-label">α Fractal</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">${(metrics.entropy_shift || 0).toFixed(3)}</span>
                    <span class="metric-label">ΔH Entropy</span>
                </div>
            </div>
        </div>
    `;

    // Agent participation
    if (agents.length > 0) {
        const agentTurns = {};
        turnStates.forEach(t => {
            agentTurns[t.agent_id] = (agentTurns[t.agent_id] || 0) + 1;
        });

        html += `
            <div class="analysis-section">
                <h3>Agent Participation</h3>
                ${agents.map(agent => {
                    const turns = agentTurns[agent] || 0;
                    const color = AGENT_COLORS[agent] || '#888888';
                    return `
                        <div class="agent-stat-row">
                            <span class="agent-dot" style="background: ${color}"></span>
                            <span class="agent-name">${agent}</span>
                            <span class="agent-turns">${turns} turns</span>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    elements.detailAnalysisContent.innerHTML = html;
}

function switchDetailTab(tabName) {
    // Update tab buttons
    elements.detailTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Show/hide tab content
    elements.detailDialogueTab.classList.toggle('hidden', tabName !== 'dialogue');
    elements.detailAnalysisTab.classList.toggle('hidden', tabName !== 'analysis');
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

// Sessions list click delegation
elements.sessionsList.addEventListener('click', (e) => {
    const sessionItem = e.target.closest('.session-item');
    if (sessionItem) {
        selectSession(sessionItem.dataset.sessionId);
    }
});

// Detail tabs
elements.detailTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        switchDetailTab(tab.dataset.tab);
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
