const API_BASE = "http://localhost:8000";

// CHAT LOGIC
function initChat() {
    const chatHistory = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');

    if (!chatHistory) return;

    let historyLog = [];

    function addMessage(role, content, isHtml = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-msg ${role}`;
        msgDiv.innerHTML = `<div class="msg-bubble">${isHtml ? content : marked.parse(content)}</div>`;
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return msgDiv;
    }

    async function sendQuestion() {
        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = '';
        addMessage('user', text);

        const currentHistory = [...historyLog];

        // Add to log (for simple state logic, API handles strict formats)
        historyLog.push({ role: 'user', content: text });

        const assistantMsgDiv = addMessage('assistant', '<div class="flex items-center gap-2 text-gray-500 font-medium text-[13px]"><i class="fa-solid fa-circle-notch fa-spin text-[#2d4a36]"></i> Thinking...</div>', true);
        const bubble = assistantMsgDiv.querySelector('.msg-bubble');

        try {
            const res = await fetch(`${API_BASE}/query/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: text, history: currentHistory })
            });

            if (!res.ok) throw new Error("API error");

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let fullText = "";
            let isFirstToken = true;
            let finalContext = "";
            let finalRoute = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));
                            if (data.token) {
                                if (isFirstToken) {
                                    bubble.innerHTML = "";
                                    isFirstToken = false;
                                }
                                fullText += data.token;
                                bubble.innerHTML = marked.parse(fullText);
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            }
                            if (data.done) {
                                if (isFirstToken) bubble.innerHTML = "";
                                finalContext = data.context || "";
                                finalRoute = data.route || "";
                                historyLog.push({ role: 'assistant', content: fullText });
                            }
                        } catch (e) { }
                    }
                }
            }

            if (finalContext && finalContext !== "No context available.") {
                const contextId = 'ctx-' + Date.now();
                const contextDiv = document.createElement('div');
                contextDiv.className = 'msg-context mt-2 pt-2 border-t border-gray-200';
                contextDiv.innerHTML = `
                    <div class="flex items-center gap-2 text-[11px] text-gray-500 mb-1 cursor-pointer select-none" onclick="document.getElementById('${contextId}').classList.toggle('hidden'); this.querySelector('.ctx-arrow').classList.toggle('fa-chevron-right'); this.querySelector('.ctx-arrow').classList.toggle('fa-chevron-down');">
                        <i class="fa-solid fa-chevron-right ctx-arrow text-[9px]"></i>
                        <i class="fa-solid fa-file-alt"></i>
                        <span class="font-medium">Retrieved Context</span>
                        ${finalRoute ? `<span class="px-2 py-0.5 bg-gray-100 rounded-full">${finalRoute}</span>` : ''}
                    </div>
                    <div id="${contextId}" class="hidden text-[12px] text-gray-600 bg-gray-50 rounded-lg p-3 max-h-48 overflow-y-auto mt-1">${marked.parse(finalContext)}</div>
                `;
                bubble.appendChild(contextDiv);
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        } catch (e) {
            bubble.innerHTML = "<span class='status-failure'>Failed to connect to API</span>";
        }
    }

    sendBtn.addEventListener('click', sendQuestion);
    chatInput.addEventListener('keypress', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    clearBtn?.addEventListener('click', () => {
        chatHistory.innerHTML = '';
        historyLog = [];
    });

    // File Upload / Ingestion Logic
    const fileUpload = document.getElementById('file-upload');
    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const statusText = document.getElementById('upload-status-text');

    if (fileUpload) {
        fileUpload.addEventListener('change', async (e) => {
            if (!e.target.files.length) return;
            const files = e.target.files;
            const isBatch = files.length > 1;

            progressContainer.classList.remove('hidden');
            progressBar.className = 'bg-brand-text h-1.5 rounded-full transition-all duration-300';
            progressBar.style.width = '20%';
            statusText.innerText = isBatch ? `Submitting ${files.length} files...` : 'Submitting to Celery...';

            const formData = new FormData();
            if (isBatch) {
                for (let i = 0; i < files.length; i++) {
                    formData.append("files", files[i]);
                }
            } else {
                formData.append("file", files[0]);
            }

            try {
                const endpoint = isBatch ? `${API_BASE}/ingest/batch` : `${API_BASE}/ingest/async`;
                const res = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (!res.ok) throw new Error(data.detail || "Upload failed");

                const taskId = data.task_id;
                statusText.innerText = `Processing (Task: ${taskId.substring(0, 8)}...)`;
                progressBar.style.width = '50%';

                let pollCount = 0;
                const pollInterval = setInterval(async () => {
                    pollCount++;
                    try {
                        const statusRes = await fetch(`${API_BASE}/ingest/status/${taskId}`);
                        const statusData = await statusRes.json();

                        if (statusData.status === 'SUCCESS') {
                            clearInterval(pollInterval);
                            progressBar.style.width = '100%';
                            const storedStr = statusData.result ? (statusData.result.stored || statusData.result.total_stored || statusData.result.total_success || 'All') : 'All';
                            statusText.innerText = `Completed! ${storedStr} vectors stored.`;
                            setTimeout(() => progressContainer.classList.add('hidden'), 5000);
                            fileUpload.value = "";
                        } else if (statusData.status === 'FAILURE') {
                            clearInterval(pollInterval);
                            progressBar.className = 'bg-red-500 h-1.5 rounded-full transition-all duration-300';
                            statusText.innerText = 'Failed: ' + (statusData.info || 'Unknown error');
                            setTimeout(() => progressContainer.classList.add('hidden'), 5000);
                            fileUpload.value = "";
                        } else {
                            const w = Math.min(90, 50 + (pollCount * 2));
                            progressBar.style.width = `${w}%`;
                        }
                    } catch (err) {
                        console.error("Polling error", err);
                    }
                }, 1000);

            } catch (e) {
                progressBar.className = 'bg-red-500 h-1.5 rounded-full transition-all duration-300';
                statusText.innerText = `Error: ${e.message}`;
                setTimeout(() => progressContainer.classList.add('hidden'), 5000);
                fileUpload.value = "";
            }
        });
    }
}

// DASHBOARD LOGIC
async function loadChart(url, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error("Failed");
        const data = await res.json();

        if (data.is_bvc) {
            container.innerHTML = `<div style="text-align:center; padding: 40px; background: #f8f9fa; border-radius: 8px;">
                <h3 style="color: #333;">${data.latest.name || data.ticker}</h3>
                <p style="font-size: 36px; font-weight: bold; color: #28a745; margin: 20px 0;">${parseFloat(data.latest.last_price || 0).toFixed(2)} MAD</p>
                <p style="font-size: 18px; color: ${data.latest.variation_pct >= 0 ? '#28a745' : '#dc3545'};">
                    ${data.latest.variation_pct >= 0 ? '▲' : '▼'} ${Math.abs(parseFloat(data.latest.variation_pct || 0)).toFixed(2)}%
                </p>
            </div>`;
            return data;
        }

        container.innerHTML = '<div style="text-align:center; padding: 20px;"><i class="fa-solid fa-spinner fa-spin"></i> Loading...</div>';
        Plotly.newPlot(containerId, data.figure.data, data.figure.layout);
        return data;
    } catch (err) {
        container.innerHTML = `<span class='status-failure'>Error loading data</span>`;
    }
}

function initDashboard() {
    if (!document.getElementById('stock-chart-container')) return;

    const loadPrices = async () => {
        const ticker = document.getElementById('ticker-input').value || 'AAPL';
        const days = document.getElementById('days-slider').value;
        const data = await loadChart(`/api/dashboard/prices?ticker=${ticker}&days=${days}`, 'stock-chart-container');

        if (data && data.latest) {
            if (data.is_bvc) {
                const v = data.latest.variation_pct;
                const varSign = v > 0 ? '+' : '';
                document.getElementById('metric-close').innerHTML = `<strong>${(data.latest.last_price || 0).toFixed(2)} MAD</strong>`;
                document.getElementById('metric-open').innerText = `${(data.latest.open || 0).toFixed(2)} MAD`;
                document.getElementById('metric-high').innerText = `${(data.latest.high || 0).toFixed(2)} MAD`;
                document.getElementById('metric-vol').innerHTML = `<span style="color: ${v >= 0 ? '#28a745' : '#dc3545'};">${varSign}${v || 0}%</span>`;
            } else {
                document.getElementById('metric-close').innerText = `$${(data.latest.close || 0).toFixed(2)}`;
                document.getElementById('metric-open').innerText = `$${(data.latest.open || 0).toFixed(2)}`;
                document.getElementById('metric-high').innerText = `$${(data.latest.high || 0).toFixed(2)}`;
                document.getElementById('metric-vol').innerText = (data.latest.volume || 0).toLocaleString();
            }
        }
    };

    document.getElementById('refresh-prices')?.addEventListener('click', loadPrices);
    document.getElementById('days-slider')?.addEventListener('change', loadPrices);

    document.getElementById('refresh-financials')?.addEventListener('click', () => {
        const comp = document.getElementById('company-input').value;
        loadChart(`/api/dashboard/financials?company=${comp}`, 'financials-chart');
    });

    // BVC Live Data
    const refreshBvcBtn = document.getElementById('refresh-bvc');
    const bvcStatus = document.getElementById('bvc-status');
    const bvcContainer = document.getElementById('bvc-table-container');

    if (refreshBvcBtn) {
        const loadBvcData = async () => {
            bvcStatus.innerText = 'Loading BVC data...';
            try {
                const res = await fetch(`${API_BASE}/bvc/prices`);
                if (!res.ok) throw new Error("Failed");
                const data = await res.json();

                if (!data.prices || data.prices.length === 0) {
                    bvcStatus.innerText = 'No BVC data available. Click Refresh to retry.';
                    bvcContainer.innerHTML = '';
                    return;
                }

                bvcStatus.innerText = `Loaded ${data.prices.length} stocks | Gainers: ${data.summary?.gainers || 0} | Losers: ${data.summary?.losers || 0}`;

                let tableHtml = `<table style="width:100%; border-collapse: collapse; font-size: 13px;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Ticker</th>
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Name</th>
                            <th style="padding: 10px; text-align: right; border-bottom: 2px solid #dee2e6;">Last Price</th>
                            <th style="padding: 10px; text-align: right; border-bottom: 2px solid #dee2e6;">Change %</th>
                            <th style="padding: 10px; text-align: right; border-bottom: 2px solid #dee2e6;">Open</th>
                            <th style="padding: 10px; text-align: right; border-bottom: 2px solid #dee2e6;">High</th>
                            <th style="padding: 10px; text-align: right; border-bottom: 2px solid #dee2e6;">Low</th>
                            <th style="padding: 10px; text-align: right; border-bottom: 2px solid #dee2e6;">Volume</th>
                        </tr>
                    </thead>
                    <tbody>`;

                for (const p of data.prices) {
                    const variation = p.variation_pct != null ? parseFloat(p.variation_pct) : null;
                    const varClass = variation > 0 ? 'color: #28a745;' : variation < 0 ? 'color: #dc3545;' : '';
                    const varSign = variation > 0 ? '+' : '';
                    tableHtml += `<tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px;"><strong>${p.ticker}</strong></td>
                        <td style="padding: 8px;">${p.name || '-'}</td>
                        <td style="padding: 8px; text-align: right;"><strong>${p.last_price != null ? p.last_price.toFixed(2) : '-'}</strong></td>
                        <td style="padding: 8px; text-align: right; ${varClass}">${variation != null ? varSign + variation.toFixed(2) + '%' : '-'}</td>
                        <td style="padding: 8px; text-align: right;">${p.open != null ? p.open.toFixed(2) : '-'}</td>
                        <td style="padding: 8px; text-align: right;">${p.high != null ? p.high.toFixed(2) : '-'}</td>
                        <td style="padding: 8px; text-align: right;">${p.low != null ? p.low.toFixed(2) : '-'}</td>
                        <td style="padding: 8px; text-align: right;">${p.volume != null ? parseFloat(p.volume).toLocaleString() : '-'}</td>
                    </tr>`;
                }
                tableHtml += '</tbody></table>';
                bvcContainer.innerHTML = tableHtml;
            } catch (err) {
                bvcStatus.innerText = 'Error loading BVC data: ' + err.message;
            }
        };

        refreshBvcBtn.addEventListener('click', () => {
            bvcStatus.innerText = 'Refreshing BVC data...';
            loadBvcData();
        });
        loadBvcData();
    }

    // Initial loads
    loadPrices();
    loadChart('/api/dashboard/financials', 'financials-chart');
    loadChart('/api/dashboard/macro', 'macro-chart');
    loadChart('/api/dashboard/movers', 'movers-chart');

    // Auto-refresh stock prices every 60 seconds
    setInterval(loadPrices, 60000);
}

// PIPELINE LOGIC
async function initPipeline() {
    const tableBody = document.getElementById('pipeline-tasks');
    if (!tableBody) return;

    const loadTasks = async () => {
        try {
            const res = await fetch('/api/pipeline/tasks');
            const data = await res.json();

            tableBody.innerHTML = '';

            let success = 0, fail = 0;

            data.tasks.forEach(t => {
                if (t.status === 'SUCCESS') success++;
                else if (t.status === 'FAILURE') fail++;

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><code>${(t.task_id || '').substring(0, 12)}...</code></td>
                    <td>${t.name || 'unknown'}</td>
                    <td class="${t.status === 'SUCCESS' ? 'status-success' : t.status === 'FAILURE' ? 'status-failure' : ''}">${t.status}</td>
                    <td>${t.date_done || ''}</td>
                `;
                tableBody.appendChild(tr);
            });

            document.getElementById('metric-total').innerText = data.tasks.length;
            document.getElementById('metric-success').innerText = success;
            document.getElementById('metric-fail').innerText = fail;

        } catch (e) {
            tableBody.innerHTML = '<tr><td colspan="4">Error loading tasks</td></tr>';
        }
    }

    loadTasks();
    document.getElementById('refresh-pipeline')?.addEventListener('click', loadTasks);
}


// ROUTER INIT
document.addEventListener('DOMContentLoaded', () => {
    initChat();
    initDashboard();
    initPipeline();

    // Simple tab switching for Dashboard
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');

            // Re-render plotly charts to fix size issues when they are unhidden
            window.dispatchEvent(new Event('resize'));
        });
    });
});
