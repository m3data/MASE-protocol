/**
 * MASE Sessions Browser Module
 *
 * Handles browsing and viewing past sessions with their
 * dialogue transcripts and analysis results.
 */

const SessionsBrowser = (function() {
    'use strict';

    // ========================================================================
    // Configuration
    // ========================================================================

    const API_BASE = '';

    // Agent colors (shared with app.js)
    const AGENT_COLORS = {
        luma: 'rgb(195, 160, 95)',
        elowen: 'rgb(130, 155, 130)',
        orin: 'rgb(100, 140, 160)',
        nyra: 'rgb(175, 130, 160)',
        ilya: 'rgb(115, 155, 155)',
        sefi: 'rgb(195, 140, 95)',
        tala: 'rgb(205, 110, 70)',
        human: 'rgb(180, 144, 112)',
        researcher: 'rgb(160, 160, 180)'
    };

    const BASIN_COLORS = {
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


    // ========================================================================
    // State
    // ========================================================================

    let state = {
        sessionsList: [],
        selectedSessionId: null,
        currentTab: 'dialogue'
    };


    // ========================================================================
    // DOM Elements (cached on init)
    // ========================================================================

    let elements = {};


    // ========================================================================
    // Utility Functions
    // ========================================================================

    function formatTimestamp(isoString) {
        if (!isoString) return '';
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
        div.textContent = text || '';
        return div.innerHTML;
    }


    // ========================================================================
    // API Functions
    // ========================================================================

    async function fetchSessionsList() {
        try {
            const response = await fetch(`${API_BASE}/api/sessions`);
            if (!response.ok) throw new Error('Failed to fetch sessions');
            const data = await response.json();
            state.sessionsList = data.sessions || [];
            return state.sessionsList;
        } catch (error) {
            console.error('Failed to load sessions:', error);
            state.sessionsList = [];
            throw error;
        }
    }

    async function fetchSessionDialogue(sessionId) {
        const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/dialogue`);
        if (!response.ok) throw new Error('Session not found');
        return await response.json();
    }

    async function fetchSessionAnalysis(sessionId) {
        const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/analysis`);
        if (!response.ok) throw new Error('Analysis not available');
        return await response.json();
    }


    // ========================================================================
    // Render Functions
    // ========================================================================

    function renderSessionsList() {
        const sessions = state.sessionsList;

        if (elements.sessionsCount) {
            elements.sessionsCount.textContent = `${sessions.length} session${sessions.length !== 1 ? 's' : ''}`;
        }

        if (!elements.sessionsList) return;

        if (sessions.length === 0) {
            elements.sessionsList.innerHTML = `
                <div class="sessions-empty">
                    No sessions yet. Start a dialogue in the Circle tab.
                </div>
            `;
            return;
        }

        elements.sessionsList.innerHTML = sessions.map(session => {
            const isSelected = state.selectedSessionId === session.session_id;
            const timestamp = formatTimestamp(session.timestamp);

            return `
                <div class="session-item ${isSelected ? 'selected' : ''}"
                     data-session-id="${session.session_id}">
                    <div class="session-item-header">
                        <span class="session-item-id">${session.session_id}</span>
                        <div class="session-item-meta">
                            <span class="session-item-turns">${session.n_turns} turns</span>
                            <span class="session-item-status ${session.has_analysis ? 'has-analysis' : 'no-analysis'}">
                                ${session.has_analysis ? 'analyzed' : 'pending'}
                            </span>
                        </div>
                    </div>
                    <div class="session-item-provocation">${escapeHtml(session.provocation)}</div>
                    ${timestamp ? `<div class="session-item-timestamp">${timestamp}</div>` : ''}
                </div>
            `;
        }).join('');
    }

    function renderDialogue(turns) {
        if (!elements.detailDialogueMessages) return;

        if (!turns || turns.length === 0) {
            elements.detailDialogueMessages.innerHTML = `
                <div class="sessions-empty">No dialogue content.</div>
            `;
            return;
        }

        elements.detailDialogueMessages.innerHTML = turns.map(turn => {
            const agentId = turn.agent_id || 'unknown';
            const color = AGENT_COLORS[agentId] || '#888888';
            const initials = agentId.slice(0, 2).toUpperCase();
            const content = turn.content || '';
            const extraClass = agentId === 'human' ? 'human' : agentId === 'researcher' ? 'researcher' : '';

            return `
                <div class="message ${extraClass}">
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

    function renderAnalysis(analysis) {
        if (!elements.detailAnalysisContent) return;

        if (!analysis) {
            elements.detailAnalysisContent.innerHTML = `
                <div class="sessions-empty">No analysis available.</div>
            `;
            return;
        }

        const metrics = analysis.semantic_metrics || {};
        const basinSequence = analysis.basin_sequence || [];
        const basinDistribution = analysis.basin_distribution || {};
        const agents = analysis.agents || [];
        const turnStates = analysis.turn_states || [];

        // Compute dominant basin info
        const dominantBasin = analysis.dominant_basin || 'Unknown';
        const dominantPct = ((analysis.dominant_basin_percentage || 0) * 100).toFixed(0);
        const transitionCount = analysis.transition_count || 0;

        // Compute coherence pattern
        const patterns = analysis.coherence_pattern_distribution || {};
        const breathing = patterns.breathing || 0;
        const transitional = patterns.transitional || 0;
        const locked = patterns.locked || 0;
        let coherencePattern = 'Transitional';
        if (breathing > transitional && breathing > locked) coherencePattern = 'Breathing';
        else if (locked > transitional && locked > breathing) coherencePattern = 'Locked';

        let html = '';

        // Summary card - richer version
        html += `
            <div class="analysis-section">
                <h3>Summary</h3>
                <div class="summary-card">
                    <div class="summary-stat">
                        <span class="stat-value">${turnStates.length}</span>
                        <span class="stat-label">Turns</span>
                    </div>
                    <div class="summary-stat">
                        <span class="stat-value">${dominantBasin}</span>
                        <span class="stat-label">Dominant Basin (${dominantPct}%)</span>
                    </div>
                    <div class="summary-stat">
                        <span class="stat-value">${coherencePattern}</span>
                        <span class="stat-label">Coherence Pattern</span>
                    </div>
                    <div class="summary-stat">
                        <span class="stat-value">${transitionCount}</span>
                        <span class="stat-label">Basin Transitions</span>
                    </div>
                </div>
            </div>
        `;

        // Basin trajectory with legend
        if (basinSequence.length > 0) {
            const bars = basinSequence.map((basin, i) => {
                const color = BASIN_COLORS[basin] || '#888888';
                return `<div class="basin-bar" style="background: ${color}" title="Turn ${i + 1}: ${basin}"></div>`;
            }).join('');

            const legendItems = Object.entries(basinDistribution).map(([basin, count]) => {
                const color = BASIN_COLORS[basin] || '#888888';
                return `<span class="legend-item">
                    <span class="legend-dot" style="background: ${color}"></span>
                    ${basin}: ${count}
                </span>`;
            }).join('');

            html += `
                <div class="analysis-section">
                    <h3>Basin Trajectory</h3>
                    <div class="basin-timeline">${bars}</div>
                    <div class="basin-legend">${legendItems}</div>
                </div>
            `;
        }

        // Semantic metrics - expanded
        html += `
            <div class="analysis-section">
                <h3>Metrics</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">${(metrics.alpha || 0).toFixed(2)}</div>
                        <div class="metric-label">DFA α</div>
                        <div class="metric-desc">Long-range correlation (0.5=noise, 1.0=pink)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(metrics.curvature || 0).toFixed(2)}</div>
                        <div class="metric-label">Semantic Curvature</div>
                        <div class="metric-desc">Trajectory complexity</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(metrics.entropy_shift || 0).toFixed(2)}</div>
                        <div class="metric-label">Entropy Shift</div>
                        <div class="metric-desc">Semantic reorganization</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(analysis.voice_distinctiveness || 0).toFixed(2)}</div>
                        <div class="metric-label">Voice Distinctiveness</div>
                        <div class="metric-desc">Agent differentiation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(analysis.inquiry_vs_mimicry_ratio || 0).toFixed(2)}</div>
                        <div class="metric-label">Inquiry Ratio</div>
                        <div class="metric-desc">Inquiry vs mimicry balance</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(analysis.semantic_velocity_mean || 0).toFixed(2)}</div>
                        <div class="metric-label">Semantic Velocity</div>
                        <div class="metric-desc">Average semantic motion</div>
                    </div>
                </div>
            </div>
        `;

        // Integrity section (if available)
        if (analysis.integrity_score !== undefined) {
            const integrityLabel = analysis.integrity_label || 'unknown';
            const integrityClass = `integrity-${integrityLabel}`;
            html += `
                <div class="analysis-section">
                    <h3>Trajectory Integrity</h3>
                    <div class="integrity-display">
                        <span class="integrity-score">${(analysis.integrity_score || 0).toFixed(2)}</span>
                        <span class="integrity-label ${integrityClass}">${integrityLabel}</span>
                    </div>
                    <div class="integrity-desc">
                        ${integrityLabel === 'fragmented' ? 'Low memory retention — chaotic trajectory' :
                          integrityLabel === 'living' ? 'Healthy balance — adaptive coherence' :
                          integrityLabel === 'rigid' ? 'High memory retention — locked trajectory' :
                          'Integrity not computed'}
                    </div>
                </div>
            `;
        }

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


    // ========================================================================
    // UI Actions
    // ========================================================================

    async function loadSessions() {
        try {
            await fetchSessionsList();
            renderSessionsList();
        } catch (error) {
            if (elements.sessionsList) {
                elements.sessionsList.innerHTML = `
                    <div class="sessions-empty">Failed to load sessions.</div>
                `;
            }
        }
    }

    async function selectSession(sessionId) {
        state.selectedSessionId = sessionId;

        // Update list selection
        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.sessionId === sessionId);
        });

        // Show detail panel
        if (elements.sessionDetailEmpty) {
            elements.sessionDetailEmpty.classList.add('hidden');
        }
        if (elements.sessionDetailContent) {
            elements.sessionDetailContent.classList.remove('hidden');
        }

        // Load and display session data
        try {
            const dialogueData = await fetchSessionDialogue(sessionId);

            // Update header
            if (elements.detailSessionId) {
                elements.detailSessionId.textContent = sessionId;
            }
            if (elements.detailTimestamp) {
                elements.detailTimestamp.textContent = formatTimestamp(dialogueData.start_time);
            }
            if (elements.detailProvocation) {
                elements.detailProvocation.textContent = dialogueData.provocation || '';
            }

            // Render dialogue
            renderDialogue(dialogueData.turns);

            // Try to load analysis
            try {
                const analysisData = await fetchSessionAnalysis(sessionId);
                renderAnalysis(analysisData);
            } catch {
                renderAnalysis(null);
            }

        } catch (error) {
            console.error('Failed to load session:', error);
            if (elements.detailDialogueMessages) {
                elements.detailDialogueMessages.innerHTML = `
                    <div class="sessions-empty">Failed to load session.</div>
                `;
            }
        }
    }

    function switchTab(tabName) {
        state.currentTab = tabName;

        // Update tab buttons
        if (elements.detailTabs) {
            elements.detailTabs.forEach(tab => {
                tab.classList.toggle('active', tab.dataset.tab === tabName);
            });
        }

        // Show/hide tab content
        if (elements.detailDialogueTab) {
            elements.detailDialogueTab.classList.toggle('hidden', tabName !== 'dialogue');
        }
        if (elements.detailAnalysisTab) {
            elements.detailAnalysisTab.classList.toggle('hidden', tabName !== 'analysis');
        }
    }


    // ========================================================================
    // Event Binding
    // ========================================================================

    function bindEvents() {
        // Sessions list click delegation
        if (elements.sessionsList) {
            elements.sessionsList.addEventListener('click', (e) => {
                const sessionItem = e.target.closest('.session-item');
                if (sessionItem) {
                    selectSession(sessionItem.dataset.sessionId);
                }
            });
        }

        // Detail tabs
        if (elements.detailTabs) {
            elements.detailTabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    switchTab(tab.dataset.tab);
                });
            });
        }
    }


    // ========================================================================
    // Public API
    // ========================================================================

    function init() {
        // Cache DOM elements
        elements = {
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

        bindEvents();
    }

    // Return public interface
    return {
        init: init,
        loadSessions: loadSessions,
        selectSession: selectSession,
        switchTab: switchTab
    };

})();
