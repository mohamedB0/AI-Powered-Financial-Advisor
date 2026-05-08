const API_BASE = "http://localhost:8000";

// ── Keyword Highlighter ───────────────────────────────────────────────────────
// Highlights query keywords inside retrieved context blockquotes
const STOPWORDS = new Set([
    // English
    "a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","must",
    "and","or","but","if","in","on","at","to","for","of","with","by","from",
    "about","what","when","where","who","why","how","which","that","this","these",
    "those","i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","can","not","no","yes","get","give",
    // French
    "le","la","les","un","une","des","du","de","et","ou","en","dans","sur","sous",
    "pour","par","avec","sans","que","qui","quoi","dont","où","comment","quand",
    "quel","quelle","quels","quelles","est","sont","était","être","avoir","je",
    "tu","il","elle","nous","vous","ils","elles","me","te","se","lui","leur",
    "ce","cet","cette","ces","mon","ton","son","ma","ta","sa","notre","votre",
    // Arabic particles
    "في","من","على","إلى","عن","هل","ما","مع","لا","هذا","هذه","ذلك",
]);

/**
 * Walk all text nodes inside el and wrap words matching any keyword with <mark>.
 */
function highlightKeywordsInContext(containerEl, question) {
    if (!containerEl || !question) return;

    // Extract meaningful keywords (length ≥ 3, not a stopword)
    const words = question
        .replace(/[?!.,;:«»""''()\[\]{}]/g, " ")
        .split(/\s+/)
        .map(w => w.trim().toLowerCase())
        .filter(w => w.length >= 3 && !STOPWORDS.has(w));

    if (words.length === 0) return;

    // Build a single regex alternation, escape special chars
    const escaped = words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const pattern = new RegExp(`(${escaped.join("|")})`, "gi");

    // Only highlight inside blockquote paragraphs (the chunk text)
    const targets = containerEl.querySelectorAll("blockquote p, blockquote");

    targets.forEach(el => {
        // Walk text nodes to avoid breaking child elements (links, code, etc.)
        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
        const textNodes = [];
        let node;
        while ((node = walker.nextNode())) textNodes.push(node);

        textNodes.forEach(tn => {
            if (!pattern.test(tn.nodeValue)) return;
            pattern.lastIndex = 0; // reset regex state

            const span = document.createElement("span");
            span.innerHTML = tn.nodeValue.replace(pattern,
                m => `<mark class="ctx-highlight">${m}</mark>`
            );
            tn.parentNode.replaceChild(span, tn);
        });
    });
}

// ── GLOBAL SETTINGS ─────────────────────────────────────────────────────────
let appSettings = JSON.parse(localStorage.getItem('ai_settings')) || {
    persona: 'Balanced',
    depth: 3,
    apiKey: '',
    focusMode: 'New Chat'
};

// CHAT LOGIC
function initChat() {
    const chatHistory = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');

    if (!chatHistory) return;

    // Intercept clicks on links inside chat bubbles to ensure they open in a new tab
    document.addEventListener('click', function(e) {
        const link = e.target.closest('.msg-bubble a');
        if (link && link.href && !link.href.startsWith('javascript:')) {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        }
    });

    let historyLog = [];

    function addMessage(role, content, isHtml = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `flex w-full mt-4 mb-2 gap-4 ${role === 'user' ? 'justify-end' : 'justify-start'}`;
        
        const avatarStr = role === 'user' 
            ? `<div class="shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 shadow-sm border border-blue-200 mt-1"><i class="fa-solid fa-user text-xs"></i></div>`
            : `<div class="shrink-0 w-8 h-8 rounded-full bg-brand-bg flex items-center justify-center text-brand-text shadow-[0_2px_8px_rgba(0,0,0,0.06)] border border-[#d2ddd5] mt-1"><i class="fa-solid fa-chart-line text-[13px]"></i></div>`;
        
        const bubbleColor = role === 'user' 
            ? 'bg-blue-600 text-white shadow-md border border-blue-700' 
            : 'bg-white text-gray-800 shadow-[0_4px_20px_rgba(0,0,0,0.04)] border border-gray-100';
            
        // Use Tailwind typography (prose) to flawlessly render tables, bold, and bullet points from markdown
        const proseClass = role === 'user' ? 'prose-invert text-white' : 'prose-slate text-gray-700';

        msgDiv.innerHTML = `
            ${role === 'assistant' ? avatarStr : ''}
            <div class="relative max-w-[85%] rounded-[20px] px-5 py-4 ${bubbleColor}">
                <div class="msg-bubble prose prose-sm ${proseClass} max-w-none leading-[1.65]">${isHtml ? content : marked.parse(content)}</div>
            </div>
            ${role === 'user' ? avatarStr : ''}
        `;
        
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
                body: JSON.stringify({ 
                    question: text, 
                    history: currentHistory,
                    persona: appSettings.persona,
                    top_k: parseInt(appSettings.depth),
                    api_key: appSettings.apiKey,
                    focus_mode: appSettings.focusMode
                })
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

                                // Add Export Actions
                                const actionsDiv = document.createElement('div');
                                actionsDiv.className = "flex items-center gap-3 mt-3 pt-3 border-t border-gray-100 text-[12px] text-gray-400 opacity-60 hover:opacity-100 transition-opacity";

                                const copyBtn = document.createElement('button');
                                copyBtn.className = "hover:text-gray-800 flex items-center gap-1 transition-colors outline-none";
                                copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
                                copyBtn.onclick = () => {
                                    navigator.clipboard.writeText(fullText);
                                    copyBtn.innerHTML = '<i class="fa-solid fa-check text-green-500"></i> Copied!';
                                    setTimeout(() => copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy', 2000);
                                };

                                const dlBtn = document.createElement('button');
                                dlBtn.className = "hover:text-gray-800 flex items-center gap-1 transition-colors outline-none";
                                dlBtn.innerHTML = '<i class="fa-solid fa-download"></i> Download .md';
                                dlBtn.onclick = () => {
                                    const a = document.createElement('a');
                                    a.href = URL.createObjectURL(new Blob([fullText], {type: 'text/markdown'}));
                                    a.download = 'Financial_Analysis_' + Date.now() + '.md';
                                    a.click();
                                };

                                const sep = document.createElement('div');
                                sep.className = "w-px h-3 bg-gray-300";

                                actionsDiv.appendChild(copyBtn);
                                actionsDiv.appendChild(sep);
                                actionsDiv.appendChild(dlBtn);

                                const bubbleWrap = assistantMsgDiv.querySelector('.relative');
                                if (bubbleWrap) bubbleWrap.appendChild(actionsDiv);
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
                    <div id="${contextId}" class="hidden text-[12px] text-gray-600 bg-gray-50 rounded-lg p-3 max-h-64 overflow-y-auto mt-1">${marked.parse(finalContext)}</div>
                `;
                bubble.appendChild(contextDiv);

                // Highlight keywords from the user's question inside the blockquotes
                const ctxEl = contextDiv.querySelector(`#${contextId}`);
                highlightKeywordsInContext(ctxEl, text);

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

    // Handle Focus Mode Tabs UI
    const focusTabs = document.querySelectorAll('.focus-tab');
    focusTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active from all
            focusTabs.forEach(t => {
                t.classList.remove('active', 'border-gray-900', 'text-gray-900', 'font-semibold');
                t.classList.add('border-transparent', 'text-gray-500', 'font-medium');
            });
            // Add active to selected
            tab.classList.remove('border-transparent', 'text-gray-500', 'font-medium');
            tab.classList.add('active', 'border-gray-900', 'text-gray-900', 'font-semibold');
            
            // Set global focus mode
            appSettings.focusMode = tab.dataset.mode;
        });
    });

    const dlReportBtn = document.getElementById('download-report-btn');
    dlReportBtn?.addEventListener('click', () => {
        if (historyLog.length === 0) {
            alert('No conversation history to export.');
            return;
        }

        let reportStr = "# AI Financial Advisory Session Report\n\n";
        reportStr += `*Generated on: ${new Date().toLocaleString()}*\n\n---\n\n`;

        for (const msg of historyLog) {
            if (msg.role === 'user') {
                reportStr += `### 🧑 Question:\n${msg.content}\n\n`;
            } else if (msg.role === 'assistant') {
                reportStr += `**🤖 Advisor Analysis:**\n${msg.content}\n\n---\n\n`;
            }
        }

        const blob = new Blob([reportStr], { type: 'text/markdown' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `Financial_Session_Report_${Date.now()}.md`;
        a.click();
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
                statusText.innerText = `Queued (Task: ${taskId.substring(0, 8)}...)`;
                progressBar.style.width = '15%';

                const detailText = document.getElementById('upload-detail-text');
                const stepOrder = ['loading', 'ocr', 'chunking', 'embedding', 'storing', 'done'];

                function highlightStep(currentStep) {
                    document.querySelectorAll('.step-indicator').forEach(el => {
                        const s = el.dataset.step;
                        const idx = stepOrder.indexOf(s);
                        const curIdx = stepOrder.indexOf(currentStep);
                        if (idx < curIdx) {
                            el.classList.remove('text-gray-400', 'text-brand-text');
                            el.classList.add('text-green-500');
                        } else if (idx === curIdx) {
                            el.classList.remove('text-gray-400', 'text-green-500');
                            el.classList.add('text-brand-text');
                        } else {
                            el.classList.remove('text-brand-text', 'text-green-500');
                            el.classList.add('text-gray-400');
                        }
                    });
                }

                let pollCount = 0;
                const pollInterval = setInterval(async () => {
                    pollCount++;
                    try {
                        const statusRes = await fetch(`${API_BASE}/ingest/status/${taskId}`);
                        const statusData = await statusRes.json();

                        if (statusData.status === 'PROGRESS') {
                            const pct = statusData.percent || Math.min(90, 15 + (pollCount * 3));
                            progressBar.style.width = `${pct}%`;
                            statusText.innerText = statusData.step ? statusData.step.charAt(0).toUpperCase() + statusData.step.slice(1) : 'Processing...';
                            if (detailText) detailText.innerText = statusData.detail || '';
                            if (statusData.step) highlightStep(statusData.step);
                        } else if (statusData.status === 'SUCCESS') {
                            clearInterval(pollInterval);
                            progressBar.style.width = '100%';
                            highlightStep('done');
                            const r = statusData.result || {};
                            const storedStr = r.stored || r.total_stored || r.total_success || 'All';
                            const ocrBadge = r.ocr_used ? ' (OCR+Vision)' : '';
                            statusText.innerText = `Done! ${storedStr} vectors stored${ocrBadge}`;
                            if (detailText) detailText.innerText = r.filename ? `${r.filename} → ${r.chunks || '?'} chunks` : '';
                            setTimeout(() => progressContainer.classList.add('hidden'), 6000);
                            fileUpload.value = "";
                        } else if (statusData.status === 'FAILURE') {
                            clearInterval(pollInterval);
                            progressBar.className = 'bg-red-500 h-1.5 rounded-full transition-all duration-300';
                            statusText.innerText = 'Failed: ' + (statusData.info || 'Unknown error');
                            if (detailText) detailText.innerText = '';
                            setTimeout(() => progressContainer.classList.add('hidden'), 6000);
                            fileUpload.value = "";
                        } else {
                            const w = Math.min(90, 15 + (pollCount * 2));
                            progressBar.style.width = `${w}%`;
                        }
                    } catch (err) {
                        console.error("Polling error", err);
                    }
                }, 1500);

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

// PIPELINE LOGIC (Chat Document Uploads Only)
async function initPipeline() {
    const tableBody = document.getElementById('pipeline-tasks');
    if (!tableBody) return;

    const loadTasks = async () => {
        try {
            const res = await fetch('/api/pipeline/tasks');
            const data = await res.json();

            tableBody.innerHTML = '';

            if (!data.tasks || data.tasks.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="4" class="p-6 text-center text-gray-400"><i class="fa-solid fa-folder-open mr-2"></i>No documents uploaded yet. Use the chat to ingest files.</td></tr>';
                document.getElementById('metric-total').innerText = '0';
                document.getElementById('metric-success').innerText = '0';
                document.getElementById('metric-fail').innerText = '0';
                return;
            }

            let success = 0, fail = 0;

            data.tasks.forEach(t => {
                if (t.status === 'SUCCESS') success++;
                else if (t.status === 'FAILURE') fail++;

                // Try to extract filename from the result JSON
                let docName = 'Document';
                try {
                    const result = typeof t.result === 'string' ? JSON.parse(t.result) : t.result;
                    if (result && result.filename) docName = result.filename;
                } catch (e) {}

                const statusClass = t.status === 'SUCCESS'
                    ? 'bg-green-50 text-green-700 border border-green-200'
                    : t.status === 'FAILURE'
                    ? 'bg-red-50 text-red-600 border border-red-200'
                    : 'bg-yellow-50 text-yellow-700 border border-yellow-200';

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="p-3"><code class="text-gray-500 text-[12px]">${(t.task_id || '').substring(0, 12)}…</code></td>
                    <td class="p-3 font-medium text-gray-800"><i class="fa-solid fa-file-lines text-gray-400 mr-1.5"></i>${docName}</td>
                    <td class="p-3"><span class="px-2.5 py-0.5 rounded-full text-[11px] font-semibold ${statusClass}">${t.status}</span></td>
                    <td class="p-3 text-gray-500">${t.date_done || '—'}</td>
                `;
                tableBody.appendChild(tr);
            });

            document.getElementById('metric-total').innerText = data.tasks.length;
            document.getElementById('metric-success').innerText = success;
            document.getElementById('metric-fail').innerText = fail;

        } catch (e) {
            tableBody.innerHTML = '<tr><td colspan="4" class="p-4 text-center text-gray-400">Error loading tasks</td></tr>';
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

    // SETTINGS MODAL LOGIC
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const settingsContent = document.getElementById('settings-modal-content');
    const closeBtn = document.getElementById('close-settings');
    const cancelBtn = document.getElementById('cancel-settings');
    const saveBtn = document.getElementById('save-settings');

    const selectPersona = document.getElementById('setting-persona');
    const sliderDepth = document.getElementById('setting-depth');
    const depthVal = document.getElementById('depth-val');
    const inputApiKey = document.getElementById('setting-api-key');

    // Update slider value text dynamically
    sliderDepth?.addEventListener('input', (e) => {
        if(depthVal) depthVal.innerText = e.target.value;
    });

    function openSettings() {
        if (!settingsModal) return;
        // Load current
        selectPersona.value = appSettings.persona;
        sliderDepth.value = appSettings.depth;
        if(depthVal) depthVal.innerText = appSettings.depth;
        inputApiKey.value = appSettings.apiKey;

        settingsModal.classList.remove('hidden');
        setTimeout(() => {
            settingsModal.classList.remove('opacity-0');
            settingsContent.classList.remove('scale-95');
            settingsContent.classList.add('scale-100');
        }, 10);
    }

    function closeSettings() {
        settingsContent.classList.remove('scale-100');
        settingsContent.classList.add('scale-95');
        setTimeout(() => {
            settingsModal.classList.add('hidden');
        }, 200);
    }

    settingsBtn?.addEventListener('click', openSettings);
    closeBtn?.addEventListener('click', closeSettings);
    cancelBtn?.addEventListener('click', closeSettings);
    settingsModal?.addEventListener('click', (e) => {
        if (e.target === settingsModal) closeSettings();
    });

    saveBtn?.addEventListener('click', () => {
        // Save to global and local storage
        appSettings.persona = selectPersona.value;
        appSettings.depth = sliderDepth.value;
        appSettings.apiKey = inputApiKey.value.trim();
        
        localStorage.setItem('ai_settings', JSON.stringify(appSettings));
        closeSettings();

        // Optional UX notification
        const oldHtml = saveBtn.innerHTML;
        saveBtn.innerHTML = '<i class="fa-solid fa-check text-green-300 mr-1.5"></i> Saved!';
        setTimeout(() => saveBtn.innerHTML = oldHtml, 2000);
    });
});
