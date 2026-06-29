// API_BASE is defined in auth.js (loaded first)

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
    focusMode: 'New Chat',
    collection: 'all',
    detail: 'balanced'
};
// Ensure collection exists for users with old cached settings
if (!appSettings.collection) appSettings.collection = 'all';

// Staged image for vision analysis (global so cancel button can access it)
let _stagedImage = null; // {file, dataUrl}

// Last question for retry functionality
let _lastQuestion = "";

// ── VOICE INPUT (Speech-to-Text) ─────────────────────────────────────────────
let _mediaRecorder = null;
let _audioChunks = [];
let _isRecording = false;

async function toggleVoiceRecording() {
    if (_isRecording) {
        stopVoiceRecording();
    } else {
        await startVoiceRecording();
    }
}

async function startVoiceRecording() {
    const micBtn = document.getElementById('mic-btn');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        _audioChunks = [];
        _mediaRecorder = new MediaRecorder(stream, {
            mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm'
        });

        _mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) _audioChunks.push(e.data);
        };

        _mediaRecorder.onstop = async () => {
            // Stop all tracks to release the mic
            stream.getTracks().forEach(t => t.stop());
            await sendAudioForTranscription();
        };

        _mediaRecorder.start();
        _isRecording = true;

        // Visual feedback
        micBtn.classList.remove('text-white/50');
        micBtn.classList.add('text-red-400', 'animate-pulse');
        micBtn.title = 'Stop recording';

    } catch (err) {
        console.error('[Voice] Mic access error:', err);
        showToast('Microphone access denied. Check browser permissions.', 'error');
    }
}

function stopVoiceRecording() {
    const micBtn = document.getElementById('mic-btn');

    if (_mediaRecorder && _mediaRecorder.state !== 'inactive') {
        _mediaRecorder.stop();
    }
    _isRecording = false;

    // Reset visual
    micBtn.classList.remove('text-red-400', 'animate-pulse');
    micBtn.classList.add('text-white/50');
    micBtn.title = 'Voice input';
}

async function sendAudioForTranscription() {
    if (_audioChunks.length === 0) return;

    const blob = new Blob(_audioChunks, { type: 'audio/webm' });
    _audioChunks = [];

    // Show feedback
    const chatInput = document.getElementById('chat-input');
    chatInput.placeholder = 'Transcribing...';

    try {
        const formData = new FormData();
        formData.append('audio', blob, 'recording.webm');

        const res = await authFetch(`${API_BASE}/speech/transcribe`, {
            method: 'POST',
            body: formData,
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || 'Transcription failed');
        }

        if (data.text) {
            // Fill the chat input with transcribed text
            chatInput.value = data.text;
            chatInput.focus();
            // Auto-resize
            chatInput.style.height = 'auto';
            chatInput.style.height = chatInput.scrollHeight + 'px';
            showToast('Voice transcribed', 'success', 2000);
        }
    } catch (err) {
        console.error('[Voice] Transcription error:', err);
        showToast(err.message || 'Voice transcription failed', 'error');
    } finally {
        chatInput.placeholder = 'Ask Something';
    }
}

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
        // On first message: hide empty state, reveal chat history
        const emptyState = document.getElementById('empty-state');
        if (emptyState) emptyState.style.display = 'none';
        chatHistory.classList.remove('hidden');
        chatHistory.classList.add('flex');

        const msgDiv = document.createElement('div');
        msgDiv.className = `flex w-full mt-4 mb-2 gap-4 ${role === 'user' ? 'justify-end' : 'justify-start'}`;
        
        // User avatar: show Google profile picture if available
        let userAvatarStr;
        const currentUser = getUser();
        if (currentUser && currentUser.picture) {
            userAvatarStr = `<div class="shrink-0 w-8 h-8 rounded-full overflow-hidden shadow-sm border border-[#3a5c44] mt-1"><img src="${currentUser.picture}" alt="You" class="w-full h-full object-cover"></div>`;
        } else {
            userAvatarStr = `<div class="shrink-0 w-8 h-8 rounded-full bg-[#2d4a36] flex items-center justify-center text-emerald-200 shadow-sm border border-[#3a5c44] mt-1"><i class="fa-solid fa-user text-xs"></i></div>`;
        }

        const avatarStr = role === 'user' 
            ? userAvatarStr
            : `<div class="shrink-0 w-8 h-8 rounded-full bg-brand-bg flex items-center justify-center text-brand-text shadow-[0_2px_8px_rgba(0,0,0,0.06)] border border-[#d2ddd5] mt-1"><i class="fa-solid fa-chart-line text-[13px]"></i></div>`;
        
        const bubbleColor = role === 'user' 
            ? 'bg-[#1a2e1f] text-white shadow-md border border-[#243b28]' 
            : 'bg-white text-gray-800 shadow-[0_4px_20px_rgba(0,0,0,0.04)] border border-gray-100';

        // Asymmetric corners: sharp corner points toward the avatar
        const bubbleRadius = role === 'user'
            ? 'rounded-[20px] rounded-br-[4px]'
            : 'rounded-[20px] rounded-bl-[4px]';
            
        // Use Tailwind typography (prose) to flawlessly render tables, bold, and bullet points from markdown
        const proseClass = role === 'user' ? 'prose-invert text-white' : 'prose-slate text-gray-700';

        const timestamp = _formatMsgTime(new Date());
        const timeAlign = role === 'user' ? 'text-right' : 'text-left';

        msgDiv.innerHTML = `
            ${role === 'assistant' ? avatarStr : ''}
            <div class="relative max-w-[85%] flex flex-col">
                <div class="${bubbleRadius} px-5 py-4 ${bubbleColor}">
                    <div class="msg-bubble prose prose-sm ${proseClass} max-w-none leading-[1.65]" dir="auto">${isHtml ? content : marked.parse(content)}</div>
                </div>
                <span class="msg-timestamp ${timeAlign}">${timestamp}</span>
            </div>
            ${role === 'user' ? avatarStr : ''}
        `;
        
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return msgDiv;
    }

    async function sendQuestion() {
        const text = chatInput.value.trim();

        // If there's a staged image, send it for vision analysis
        if (_stagedImage) {
            const file = _stagedImage.file;
            const imgSrc = _stagedImage.dataUrl;
            const question = text;

            // Clear staged state
            _stagedImage = null;
            _removeImagePreview();
            chatInput.value = "";
            chatInput.style.height = '54px';
            chatInput.placeholder = "Ask Something";

            // Display user message with image thumbnail
            const userContent = `
                <div class="flex items-start gap-3">
                    <img src="${imgSrc}" alt="Uploaded image" class="w-20 h-20 rounded-lg object-cover border border-white/20 shadow-sm">
                    <div>
                        <span class="text-[11px] text-emerald-300 font-semibold uppercase tracking-wide">Image Analysis</span>
                        <p class="text-sm mt-0.5">${question || file.name}</p>
                    </div>
                </div>`;
            addMessage('user', userContent, true);

            // Show typing indicator
            const typingId = showTypingIndicator();

            try {
                const formData = new FormData();
                formData.append('file', file);
                if (question) formData.append('question', question);

                const res = await authFetch(`${API_BASE}/vision/analyze`, {
                    method: 'POST',
                    body: formData,
                });

                removeTypingIndicator(typingId);

                if (!res.ok) {
                    const err = await res.json().catch(() => ({ detail: 'Vision analysis failed' }));
                    addMessage('assistant', `**Vision Analysis Error:** ${err.detail || 'Unknown error'}`);
                    return;
                }

                const data = await res.json();
                const modelBadge = `<span class="text-[10px] text-gray-400 font-normal">(${data.model})</span>`;
                const extractionPreview = data.extraction
                    ? `<details class="mt-3 text-[12px]"><summary class="cursor-pointer text-gray-500 hover:text-gray-700 font-medium">View extracted content</summary><pre class="mt-2 p-3 bg-gray-50 rounded-lg text-gray-600 overflow-x-auto whitespace-pre-wrap text-[11px] border border-gray-100 max-h-48 overflow-y-auto">${data.extraction.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre></details>`
                    : '';
                addMessage('assistant', `**Financial Image Analysis** ${modelBadge}\n\n${data.analysis}${extractionPreview}`, true);

            } catch (err) {
                removeTypingIndicator(typingId);
                addMessage('assistant', `**Error:** ${err.message || 'Image analysis failed. Check your connection.'}`);
            }
            return;
        }

        // Normal text message
        if (!text) return;

        _lastQuestion = text;
        chatInput.value = '';
        addMessage('user', text);

        const currentHistory = [...historyLog];

        // Add to log (for simple state logic, API handles strict formats)
        historyLog.push({ role: 'user', content: text });

        // Persist user message to DB
        const convId = await ensureConversation(text);
        if (convId) saveMessage(convId, 'user', text);

        const assistantMsgDiv = addMessage('assistant',
            `<div class="typing-indicator">
                <div class="typing-wave">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
                <span class="typing-label">Searching documents...</span>
            </div>`, true);
        const bubble = assistantMsgDiv.querySelector('.msg-bubble');

        // Animate stage labels while waiting for first token
        const _stageLabels = ["Searching documents...", "Analyzing relevance...", "Generating answer..."];
        let _stageIdx = 0;
        const _stageTimer = setInterval(() => {
            _stageIdx = Math.min(_stageIdx + 1, _stageLabels.length - 1);
            const label = bubble.querySelector('.typing-label');
            if (label) label.textContent = _stageLabels[_stageIdx];
        }, 1500);

        try {
            const res = await authFetch(`${API_BASE}/query/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    question: text, 
                    history: currentHistory,
                    persona: appSettings.persona,
                    top_k: parseInt(appSettings.depth),
                    focus_mode: appSettings.focusMode,
                    collection: appSettings.collection || "all",
                    detail: appSettings.detail || "balanced"
                })
            });

            if (!res.ok) throw new Error("API error");

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let fullText = "";
            let isFirstToken = true;
            let finalContext = "";
            let finalRoute = "";
            let finalChart = null;

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
                                    clearInterval(_stageTimer);
                                    bubble.innerHTML = "";
                                    isFirstToken = false;
                                }
                                fullText += data.token;
                                // Strip any [SOURCE: ...] inline citations the LLM may still emit
                                const cleanText = fullText.replace(/\[SOURCE:[^\]]*\]/gi, '').replace(/\s{2,}/g, ' ').trimStart();
                                bubble.innerHTML = marked.parse(cleanText);
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            }
                            if (data.done) {
                                clearInterval(_stageTimer);
                                if (isFirstToken) bubble.innerHTML = "";
                                finalContext = data.context || "";
                                finalRoute = data.route || "";
                                finalChart = data.chart || null;
                                historyLog.push({ role: 'assistant', content: fullText });

                                // Persist assistant message to DB
                                if (_currentConversationId) {
                                    saveMessage(_currentConversationId, 'assistant', fullText, finalRoute);
                                    refreshHistory();
                                }

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

                                // Feedback buttons (thumbs up/down)
                                const sep2 = document.createElement('div');
                                sep2.className = "w-px h-3 bg-gray-300";
                                actionsDiv.appendChild(sep2);

                                const thumbUp = document.createElement('button');
                                thumbUp.className = "hover:text-emerald-600 flex items-center gap-1 transition-colors outline-none";
                                thumbUp.innerHTML = '<i class="fa-regular fa-thumbs-up"></i>';
                                thumbUp.title = "Good answer";
                                thumbUp.onclick = () => {
                                    thumbUp.innerHTML = '<i class="fa-solid fa-thumbs-up text-emerald-500"></i>';
                                    thumbDown.innerHTML = '<i class="fa-regular fa-thumbs-down"></i>';
                                    _sendFeedback(text, fullText, 'positive');
                                };

                                const thumbDown = document.createElement('button');
                                thumbDown.className = "hover:text-red-500 flex items-center gap-1 transition-colors outline-none";
                                thumbDown.innerHTML = '<i class="fa-regular fa-thumbs-down"></i>';
                                thumbDown.title = "Poor answer";
                                thumbDown.onclick = () => {
                                    thumbDown.innerHTML = '<i class="fa-solid fa-thumbs-down text-red-400"></i>';
                                    thumbUp.innerHTML = '<i class="fa-regular fa-thumbs-up"></i>';
                                    _sendFeedback(text, fullText, 'negative');
                                };

                                actionsDiv.appendChild(thumbUp);
                                actionsDiv.appendChild(thumbDown);

                                const bubbleWrap = assistantMsgDiv.querySelector('.relative');
                                if (bubbleWrap) bubbleWrap.appendChild(actionsDiv);

                                // ── Follow-up suggestions ────────────────────
                                const suggestions = _generateFollowUps(text, finalRoute);
                                if (suggestions.length > 0) {
                                    const sugDiv = document.createElement('div');
                                    sugDiv.className = "flex flex-wrap gap-2 mt-3 pt-2";
                                    suggestions.forEach(s => {
                                        const btn = document.createElement('button');
                                        btn.className = "text-[11px] px-3 py-1.5 rounded-full border border-gray-200 text-gray-600 hover:border-emerald-300 hover:text-emerald-700 hover:bg-emerald-50 transition-all";
                                        btn.textContent = s;
                                        btn.onclick = () => {
                                            document.getElementById('chat-input').value = s;
                                            document.getElementById('send-btn')?.click();
                                        };
                                        sugDiv.appendChild(btn);
                                    });
                                    if (bubbleWrap) bubbleWrap.appendChild(sugDiv);
                                }
                            }
                        } catch (e) { }
                    }
                }
            }

            if (finalContext && finalContext !== "No context available.") {
                // ── Inline Chart: render if backend sent chart data ───────
                if (finalChart && finalChart.plotly) {
                    const chartDiv = document.createElement('div');
                    chartDiv.className = "mt-3 rounded-lg border border-gray-200 overflow-hidden";
                    chartDiv.style.height = "320px";
                    chartDiv.id = 'chart-' + Date.now();
                    bubble.appendChild(chartDiv);
                    try {
                        Plotly.newPlot(chartDiv.id, finalChart.plotly.data, finalChart.plotly.layout, {
                            responsive: true,
                            displayModeBar: false,
                        });
                    } catch (e) {
                        console.error('[Chart] Plotly render error:', e);
                    }
                }

                // ── Source cards: extract doc references from context ───
                const sourceCards = _extractSourceCards(finalContext);
                if (sourceCards.length > 0) {
                    const cardsDiv = document.createElement('div');
                    cardsDiv.className = "flex flex-wrap gap-2 mt-3 pt-2 border-t border-gray-100";
                    sourceCards.forEach(src => {
                        const card = document.createElement('a');
                        card.className = "flex items-center gap-2 px-3 py-1.5 rounded-lg border border-gray-200 text-[11px] text-gray-600 hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-700 transition-all cursor-pointer no-underline";
                        card.title = `${src.file} — Score: ${src.score}`;
                        if (src.file.endsWith('.pdf')) {
                            card.href = `pdf-viewer:${encodeURIComponent(src.file)}${src.page ? '&page=' + src.page : ''}`;
                        }
                        card.innerHTML = `<i class="fa-regular fa-file-pdf text-red-400"></i><span class="truncate max-w-[140px]">${src.shortName}</span>${src.page ? '<span class="text-gray-400">p.' + src.page + '</span>' : ''}`;
                        cardsDiv.appendChild(card);
                    });
                    bubble.appendChild(cardsDiv);
                }

                const contextId = 'ctx-' + Date.now();
                const contextDiv = document.createElement('div');
                contextDiv.className = 'msg-context mt-2 pt-2 border-t border-gray-200';
                contextDiv.innerHTML = `
                    <div class="flex items-center gap-2 text-[11px] text-gray-500 mb-1 cursor-pointer select-none" onclick="document.getElementById('${contextId}').classList.toggle('hidden'); this.querySelector('.ctx-arrow').classList.toggle('fa-chevron-right'); this.querySelector('.ctx-arrow').classList.toggle('fa-chevron-down');">
                        <i class="fa-solid fa-chevron-right ctx-arrow text-[9px]"></i>
                        <i class="fa-solid fa-file-alt"></i>
                        <span class="font-medium">Retrieved Context</span>
                        ${finalRoute ? (() => {
                            const r = finalRoute.toLowerCase();
                            const cls = r.includes('vector') ? 'route-badge-vector'
                                      : r.includes('web')    ? 'route-badge-web'
                                      : r.includes('sql')    ? 'route-badge-sql'
                                      : r.includes('cache')  ? 'route-badge-cache'
                                      : 'route-badge-hybrid';
                            return `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold ${cls}">${finalRoute}</span>`;
                        })() : ''}
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
            clearInterval(_stageTimer);
            // If we already streamed a successful answer, the error happened in
            // post-processing (rendering context, highlights, etc.) — don't wipe
            // out the user's answer. Just log it and keep the bubble as-is.
            if (fullText && fullText.length > 0) {
                console.error('[Stream] Post-processing error (answer preserved):', e);
                return;
            }
            bubble.innerHTML = `<div class="flex flex-col gap-2">
                <span class="status-failure text-red-500 text-[13px]">Failed to connect to API</span>
                <button onclick="retryLastQuestion(this)" class="self-start text-[12px] text-emerald-600 hover:text-emerald-800 font-medium flex items-center gap-1.5 transition-colors">
                    <i class="fa-solid fa-rotate-right text-[11px]"></i> Retry
                </button>
            </div>`;
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
        chatHistory.classList.add('hidden');
        chatHistory.classList.remove('flex');
        historyLog = [];
        const emptyState = document.getElementById('empty-state');
        if (emptyState) emptyState.style.display = '';
        // Start a fresh conversation
        if (typeof startNewConversation === 'function') startNewConversation();
    });

    // ── Clear Semantic Cache ────────────────────────────────────────────────
    const clearCacheBtn = document.getElementById('clear-cache-btn');
    clearCacheBtn?.addEventListener('click', async () => {
        // Visual feedback while the request is in-flight
        const original = clearCacheBtn.innerHTML;
        clearCacheBtn.disabled = true;
        clearCacheBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Clearing...';

        try {
            const res = await authFetch(`${API_BASE}/query/cache`, { method: 'DELETE' });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            const cleared = data.cleared || 0;
            showToast(
                cleared > 0
                    ? `Cache cleared (${cleared} entr${cleared === 1 ? 'y' : 'ies'} removed)`
                    : 'Cache was already empty',
                'success', 3000
            );
        } catch (err) {
            console.error('[Cache] Clear error:', err);
            showToast(`Failed to clear cache: ${err.message || 'unknown error'}`, 'error', 4000);
        } finally {
            clearCacheBtn.innerHTML = original;
            clearCacheBtn.disabled = false;
        }
    });

    // Prompt suggestion buttons — click to fill the input
    document.querySelectorAll('.prompt-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            chatInput.value = btn.innerText.trim();
            chatInput.focus();
        });
    });

    // Handle Focus Mode Tabs UI
    const focusTabs = document.querySelectorAll('.focus-tab');
    focusTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active from all
            focusTabs.forEach(t => {
                t.classList.remove('active', 'border-[#1a2e1f]', 'text-gray-900', 'font-semibold');
                t.classList.add('border-transparent', 'text-gray-500', 'font-medium');
            });
            // Add active to selected
            tab.classList.remove('border-transparent', 'text-gray-500', 'font-medium');
            tab.classList.add('active', 'border-[#1a2e1f]', 'text-gray-900', 'font-semibold');
            
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

        // Build a styled HTML document for PDF export
        const user = getUser();
        const userName = user?.full_name || user?.email || 'User';
        const dateStr = new Date().toLocaleString('fr-FR', { dateStyle: 'long', timeStyle: 'short' });

        let messagesHtml = '';
        for (const msg of historyLog) {
            if (msg.role === 'user') {
                messagesHtml += `
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                            <div style="width: 28px; height: 28px; border-radius: 50%; background: #1a2e1f; display: flex; align-items: center; justify-content: center;">
                                <span style="color: #a7f3d0; font-size: 12px; font-weight: bold;">U</span>
                            </div>
                            <span style="font-size: 12px; font-weight: 600; color: #374151;">You</span>
                        </div>
                        <div style="background: #f3f4f6; border-radius: 12px; padding: 14px 18px; margin-left: 36px; font-size: 13px; color: #1f2937; line-height: 1.6;">
                            ${marked.parse(msg.content)}
                        </div>
                    </div>`;
            } else if (msg.role === 'assistant') {
                messagesHtml += `
                    <div style="margin-bottom: 24px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                            <div style="width: 28px; height: 28px; border-radius: 50%; background: #eef2ee; display: flex; align-items: center; justify-content: center;">
                                <span style="color: #2d4a36; font-size: 13px;">&#x1F4CA;</span>
                            </div>
                            <span style="font-size: 12px; font-weight: 600; color: #374151;">Alpha Insight</span>
                        </div>
                        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 20px; margin-left: 36px; font-size: 13px; color: #1f2937; line-height: 1.7;">
                            ${marked.parse(msg.content)}
                        </div>
                    </div>`;
            }
        }

        const reportHtml = `
            <div id="pdf-report-content" style="font-family: 'Segoe UI', sans-serif; max-width: 700px; margin: 0 auto; padding: 40px 30px;">
                <!-- Header -->
                <div style="text-align: center; margin-bottom: 36px; padding-bottom: 24px; border-bottom: 2px solid #e5e7eb;">
                    <h1 style="font-size: 24px; font-weight: 700; color: #1a2e1f; margin: 0;">
                        Alpha <span style="color: #059669; font-style: italic;">Insight</span>
                    </h1>
                    <p style="font-size: 13px; color: #6b7280; margin-top: 6px;">Financial Advisory Session Report</p>
                </div>

                <!-- Meta -->
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #6b7280; margin-bottom: 28px; padding: 12px 16px; background: #f9fafb; border-radius: 8px;">
                    <span><strong>User:</strong> ${userName}</span>
                    <span><strong>Date:</strong> ${dateStr}</span>
                    <span><strong>Messages:</strong> ${historyLog.length}</span>
                </div>

                <!-- Messages -->
                ${messagesHtml}

                <!-- Footer -->
                <div style="margin-top: 36px; padding-top: 16px; border-top: 1px solid #e5e7eb; text-align: center; font-size: 11px; color: #9ca3af;">
                    Generated by Alpha Insight — AI Financial Advisor | PFE 2026
                </div>
            </div>
        `;

        // Create hidden container
        const container = document.createElement('div');
        container.style.position = 'fixed';
        container.style.left = '-9999px';
        container.style.top = '0';
        container.innerHTML = reportHtml;
        document.body.appendChild(container);

        const element = container.querySelector('#pdf-report-content');

        // Use html2pdf.js
        const opt = {
            margin: [10, 5, 10, 5],
            filename: `Alpha_Insight_Report_${new Date().toISOString().split('T')[0]}.pdf`,
            image: { type: 'jpeg', quality: 0.95 },
            html2canvas: { scale: 2, useCORS: true, letterRendering: true },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
            pagebreak: { mode: ['avoid-all', 'css', 'legacy'] }
        };

        // Change button text during export
        const oldHtml = dlReportBtn.innerHTML;
        dlReportBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating PDF...';
        dlReportBtn.disabled = true;

        html2pdf().set(opt).from(element).save().then(() => {
            document.body.removeChild(container);
            dlReportBtn.innerHTML = '<i class="fa-solid fa-check text-green-500"></i> Downloaded!';
            setTimeout(() => { dlReportBtn.innerHTML = oldHtml; dlReportBtn.disabled = false; }, 2500);
        }).catch(err => {
            console.error('PDF export failed:', err);
            document.body.removeChild(container);
            dlReportBtn.innerHTML = oldHtml;
            dlReportBtn.disabled = false;
            // Fallback: download as markdown
            const blob = new Blob([historyLog.map(m => `${m.role === 'user' ? '### Question' : '### Answer'}\n${m.content}\n\n---\n`).join('')], { type: 'text/markdown' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `Alpha_Insight_Report_${Date.now()}.md`;
            a.click();
        });
    });

    // ── Attachment Menu (Document Upload vs Image Analysis) ───────────────────
    const imageUpload = document.getElementById('image-upload');

    if (imageUpload) {
        imageUpload.addEventListener('change', (e) => {
            if (!e.target.files.length) return;
            const file = e.target.files[0];
            imageUpload.value = "";

            // Stage the image — don't send yet. Show a preview and let user type.
            const reader = new FileReader();
            reader.onload = function (ev) {
                _stagedImage = { file, dataUrl: ev.target.result };
                _showImagePreview(file.name, ev.target.result);
                chatInput.focus();
                chatInput.placeholder = "Ask a question about this image...";
            };
            reader.readAsDataURL(file);
        });
    }

    function _showImagePreview(filename, dataUrl) {
        // Show a small preview bar above the input
        let preview = document.getElementById('image-preview-bar');
        if (!preview) {
            preview = document.createElement('div');
            preview.id = 'image-preview-bar';
            preview.className = 'w-full mb-2 flex items-center gap-3 bg-white border border-gray-200 rounded-xl px-3 py-2 shadow-sm';
            const inputArea = document.querySelector('.input-ring');
            if (inputArea && inputArea.parentElement) {
                inputArea.parentElement.insertBefore(preview, inputArea);
            }
        }
        preview.innerHTML = `
            <img src="${dataUrl}" alt="Preview" class="w-10 h-10 rounded-lg object-cover border border-gray-200">
            <div class="flex-1 min-w-0">
                <div class="text-[12px] font-semibold text-gray-700 truncate">${filename}</div>
                <div class="text-[10px] text-gray-400">Image staged — type your question and press Enter</div>
            </div>
            <button onclick="cancelStagedImage()" class="w-7 h-7 flex items-center justify-center rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors" title="Cancel">
                <i class="fa-solid fa-xmark text-sm"></i>
            </button>`;
    }

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
            const fileExt = files[0]?.name?.split('.').pop()?.toLowerCase();
            const isExcelOrCsv = ['xlsx', 'xls', 'csv', 'tsv'].includes(fileExt);
            if (isBatch) {
                statusText.innerText = `Submitting ${files.length} files...`;
            } else if (isExcelOrCsv) {
                statusText.innerText = 'Ingesting Excel/CSV...';
            } else {
                statusText.innerText = 'Submitting to Celery...';
            }

            const formData = new FormData();
            if (isBatch) {
                for (let i = 0; i < files.length; i++) {
                    formData.append("files", files[i]);
                }
            } else {
                formData.append("file", files[0]);
            }

            try {
                // Check file extensions to decide endpoint
                const fileExt = files[0]?.name?.split('.').pop()?.toLowerCase();
                const isExcelOrCsv = ['xlsx', 'xls', 'csv', 'tsv'].includes(fileExt);
                
                let endpoint;
                if (isBatch) {
                    endpoint = `${API_BASE}/ingest/batch`;
                } else if (isExcelOrCsv) {
                    // XLSX/CSV: use sync endpoint (faster, no Celery needed)
                    endpoint = `${API_BASE}/ingest/`;
                } else {
                    // PDF/DOCX: use Celery async (slower, needs background processing)
                    endpoint = `${API_BASE}/ingest/async`;
                }
                
                const res = await authFetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (!res.ok) throw new Error(data.detail || "Upload failed");

                let taskId;
                if (isExcelOrCsv) {
                    // Sync response: direct result (no Celery)
                    taskId = null;
                    progressBar.style.width = '100%';
                    const stored = data.stored || data.chunks || 0;
                    statusText.innerText = `Success! ${stored} vectors stored.`;
                    setTimeout(() => {
                        progressContainer.classList.add('hidden');
                        showToast(`${data.filename} ingested successfully!`, 'success');
                    }, 2000);
                    fileUpload.value = '';
                    // Refresh the knowledge-base collection counts
                    if (typeof initCollectionBar === 'function') initCollectionBar();
                } else {
                    // Async response: Celery task (PDF/DOCX)
                    taskId = data.task_id;
                    // ── Persist task so the progress bar survives page navigation ──
                    localStorage.setItem('active_ingest_task', JSON.stringify({
                        taskId,
                        filename: files[0]?.name || 'Document',
                        startedAt: Date.now()
                    }));
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
                            localStorage.removeItem('active_ingest_task');
                            progressBar.style.width = '100%';
                            highlightStep('done');
                            const r = statusData.result || {};
                            const storedStr = r.stored || r.total_stored || r.total_success || 'All';
                            const ocrBadge = r.ocr_used ? ' (OCR+Vision)' : '';
                            statusText.innerText = `Done! ${storedStr} vectors stored${ocrBadge}`;
                            if (detailText) detailText.innerText = r.filename ? `${r.filename} → ${r.chunks || '?'} chunks` : '';
                            showToast(`Document ingested successfully — ${storedStr} vectors stored${ocrBadge}`, 'success');
                            setTimeout(() => progressContainer.classList.add('hidden'), 6000);
                            fileUpload.value = "";
                            // Refresh the knowledge-base collection counts (My Uploads grew)
                            if (typeof initCollectionBar === 'function') initCollectionBar();
                        } else if (statusData.status === 'FAILURE') {
                            clearInterval(pollInterval);
                            localStorage.removeItem('active_ingest_task');
                            progressBar.className = 'bg-red-500 h-1.5 rounded-full transition-all duration-300';
                            statusText.innerText = 'Failed: ' + (statusData.info || 'Unknown error');
                            if (detailText) detailText.innerText = '';
                            showToast(`Ingestion failed: ${statusData.info || 'Unknown error'}`, 'error', 6000);
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

                } // End else (Celery async)

            } catch (e) {
                progressBar.className = 'bg-red-500 h-1.5 rounded-full transition-all duration-300';
                statusText.innerText = `Error: ${e.message}`;
                setTimeout(() => progressContainer.classList.add('hidden'), 5000);
                fileUpload.value = "";
            }
        });
    }

    // ── On page load: resume polling if there's an active task from a previous session ──
    const savedTask = localStorage.getItem('active_ingest_task');
    if (savedTask && progressContainer) {
        try {
            const { taskId, filename } = JSON.parse(savedTask);
            progressContainer.classList.remove('hidden');
            progressBar.style.width = '30%';
            statusText.innerText = `Resuming: ${filename || 'Document'}...`;
            const detailText = document.getElementById('upload-detail-text');
            const stepOrder = ['loading', 'ocr', 'chunking', 'embedding', 'storing', 'done'];
            function highlightStepResume(currentStep) {
                document.querySelectorAll('.step-indicator').forEach(el => {
                    const s  = el.dataset.step;
                    const idx    = stepOrder.indexOf(s);
                    const curIdx = stepOrder.indexOf(currentStep);
                    if (idx < curIdx) { el.className = el.className.replace(/text-\S+/g, ''); el.classList.add('text-green-500'); }
                    else if (idx === curIdx) { el.className = el.className.replace(/text-\S+/g, ''); el.classList.add('text-brand-text'); }
                    else { el.className = el.className.replace(/text-\S+/g, ''); el.classList.add('text-gray-400'); }
                });
            }
            let resumePollCount = 0;
            const resumeInterval = setInterval(async () => {
                resumePollCount++;
                try {
                    const statusRes  = await fetch(`${API_BASE}/ingest/status/${taskId}`);
                    const statusData = await statusRes.json();
                    if (statusData.status === 'PROGRESS') {
                        const pct = statusData.percent || Math.min(90, 30 + (resumePollCount * 3));
                        progressBar.style.width = `${pct}%`;
                        statusText.innerText = statusData.step ? statusData.step.charAt(0).toUpperCase() + statusData.step.slice(1) : 'Processing...';
                        if (detailText) detailText.innerText = statusData.detail || '';
                        if (statusData.step) highlightStepResume(statusData.step);
                    } else if (statusData.status === 'SUCCESS') {
                        clearInterval(resumeInterval);
                        localStorage.removeItem('active_ingest_task');
                        progressBar.style.width = '100%';
                        highlightStepResume('done');
                        const r = statusData.result || {};
                        const storedStr = r.stored || r.total_stored || r.total_success || 'All';
                        statusText.innerText = `Done! ${storedStr} vectors stored`;
                        if (detailText) detailText.innerText = r.filename ? `${r.filename} ingested successfully` : '';
                        setTimeout(() => progressContainer.classList.add('hidden'), 6000);
                    } else if (statusData.status === 'FAILURE') {
                        clearInterval(resumeInterval);
                        localStorage.removeItem('active_ingest_task');
                        progressBar.className = 'bg-red-500 h-1.5 rounded-full transition-all duration-300';
                        statusText.innerText = 'Failed: ' + (statusData.info || 'Unknown error');
                        setTimeout(() => progressContainer.classList.add('hidden'), 6000);
                    } else if (statusData.status === 'PENDING' && resumePollCount > 20) {
                        // Task vanished (worker was killed) — stop polling and clean up
                        clearInterval(resumeInterval);
                        localStorage.removeItem('active_ingest_task');
                        progressContainer.classList.add('hidden');
                    }
                } catch (err) { console.error('Resume poll error', err); }
            }, 2000);
        } catch (e) {
            localStorage.removeItem('active_ingest_task');
        }
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
        const collSelect = document.getElementById('setting-collection');
        if (collSelect) collSelect.value = appSettings.collection || 'all';
        const detailSelect = document.getElementById('setting-detail');
        if (detailSelect) detailSelect.value = appSettings.detail || 'balanced';

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
        const collSelect = document.getElementById('setting-collection');
        if (collSelect) appSettings.collection = collSelect.value;
        const detailSelect = document.getElementById('setting-detail');
        if (detailSelect) appSettings.detail = detailSelect.value;
        
        localStorage.setItem('ai_settings', JSON.stringify(appSettings));
        closeSettings();
        showToast('Settings saved', 'success', 2000);

        // Keep the top collection bar in sync with the settings dropdown
        if (typeof renderCollectionBar === 'function') renderCollectionBar();

        // Optional UX notification
        const oldHtml = saveBtn.innerHTML;
        saveBtn.innerHTML = '<i class="fa-solid fa-check text-green-300 mr-1.5"></i> Saved!';
        setTimeout(() => saveBtn.innerHTML = oldHtml, 2000);
    });

    // KNOWLEDGE BASE / COLLECTION SELECTOR BAR
    initCollectionBar();
});


// ── Knowledge Base / Collection Selector ────────────────────────────────────
// Lets the user pick which vector-store collection to interrogate:
//   - all            → both collections
//   - financial_data → scraped institutional reports
//   - user_data      → documents the user uploaded
// Selection is persisted in appSettings.collection and sent with every query.

let _collectionsMeta = [];

async function initCollectionBar() {
    const bar = document.getElementById('collection-bar');
    if (!bar) return;

    // Fallback metadata so the bar renders even if the API call fails
    const fallback = [
        { id: 'all',            label: 'All Sources',        description: 'Institutional reports + your uploads', icon: 'fa-layer-group',      count: null },
        { id: 'financial_data', label: 'Institutional Docs', description: 'Scraped reports (AMMC, BKAM, HCP, CDC)', icon: 'fa-building-columns', count: null },
        { id: 'user_data',      label: 'My Uploads',         description: 'Documents you ingested',                icon: 'fa-folder-open',      count: null },
    ];

    try {
        const res = await fetch(`${API_BASE}/query/collections`);
        if (res.ok) {
            const data = await res.json();
            _collectionsMeta = (data.collections && data.collections.length) ? data.collections : fallback;
        } else {
            _collectionsMeta = fallback;
        }
    } catch (e) {
        console.warn('[Collections] Could not fetch live counts, using fallback.', e);
        _collectionsMeta = fallback;
    }

    if (!appSettings.collection) appSettings.collection = 'all';
    renderCollectionBar();
}

function renderCollectionBar() {
    const bar = document.getElementById('collection-bar');
    if (!bar) return;

    const active = appSettings.collection || 'all';
    bar.innerHTML = '';

    _collectionsMeta.forEach(c => {
        const isActive = c.id === active;
        const pill = document.createElement('button');
        pill.type = 'button';
        pill.dataset.collection = c.id;
        pill.title = c.description || c.label;
        pill.className = [
            'collection-pill flex items-center gap-2 px-3 py-1.5 rounded-lg text-[12px] font-semibold',
            'whitespace-nowrap transition-all duration-200',
            isActive
                ? 'bg-[#1a2e1f] text-white shadow-sm'
                : 'text-gray-500 hover:text-[#2d4a36] hover:bg-emerald-50'
        ].join(' ');

        const countStr = '';

        pill.innerHTML = `
            <i class="fa-solid ${c.icon || 'fa-database'} text-[11px] ${isActive ? 'text-emerald-400' : 'text-gray-400'}"></i>
            <span>${c.label}</span>
            ${countStr}`;

        pill.addEventListener('click', () => setActiveCollection(c.id));
        bar.appendChild(pill);
    });

    // Update the active-collection hint text + keep settings dropdown in sync
    const hint = document.getElementById('collection-active-hint');
    const activeMeta = _collectionsMeta.find(c => c.id === active);
    if (hint && activeMeta) {
        hint.textContent = activeMeta.description || '';
    }
    const collSelect = document.getElementById('setting-collection');
    if (collSelect) collSelect.value = active;
}

function setActiveCollection(id) {
    if (appSettings.collection === id) return;
    appSettings.collection = id;
    localStorage.setItem('ai_settings', JSON.stringify(appSettings));
    renderCollectionBar();

    const meta = _collectionsMeta.find(c => c.id === id);
    showToast(`Knowledge base: ${meta ? meta.label : id}`, 'info', 2000);
}


// ── Attachment Menu (Document Upload vs Image Analysis) ──────────────────────

function toggleAttachMenu() {
    const menu = document.getElementById('attach-menu');
    if (!menu) return;
    menu.classList.toggle('hidden');
}

function triggerDocUpload() {
    toggleAttachMenu();
    document.getElementById('file-upload').click();
}

function triggerImageAnalysis() {
    toggleAttachMenu();
    document.getElementById('image-upload').click();
}

// Close attach menu when clicking outside
document.addEventListener('click', (e) => {
    const wrapper = document.getElementById('attach-menu-wrapper');
    const menu = document.getElementById('attach-menu');
    if (!wrapper || !menu) return;
    if (!wrapper.contains(e.target)) {
        menu.classList.add('hidden');
    }
});

// ── Typing Indicator Helpers ─────────────────────────────────────────────────

let _typingCounter = 0;

function showTypingIndicator() {
    const chatHistory = document.getElementById('chat-history');
    if (!chatHistory) return null;

    // Show chat history if hidden
    const emptyState = document.getElementById('empty-state');
    if (emptyState) emptyState.style.display = 'none';
    chatHistory.classList.remove('hidden');
    chatHistory.classList.add('flex');

    const id = `typing-${++_typingCounter}`;
    const div = document.createElement('div');
    div.id = id;
    div.className = 'flex w-full mt-4 mb-2 gap-4 justify-start';
    div.innerHTML = `
        <div class="shrink-0 w-8 h-8 rounded-full bg-brand-bg flex items-center justify-center text-brand-text shadow-[0_2px_8px_rgba(0,0,0,0.06)] border border-[#d2ddd5] mt-1">
            <i class="fa-solid fa-chart-line text-[13px]"></i>
        </div>
        <div class="bg-white text-gray-800 shadow-[0_4px_20px_rgba(0,0,0,0.04)] border border-gray-100 rounded-[20px] rounded-bl-[4px] px-5 py-4 max-w-[85%]">
            <div class="flex items-center gap-2">
                <div class="flex gap-1">
                    <span class="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style="animation-delay:0ms"></span>
                    <span class="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style="animation-delay:150ms"></span>
                    <span class="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style="animation-delay:300ms"></span>
                </div>
                <span class="text-[12px] text-gray-400 ml-2">Analyzing image...</span>
            </div>
        </div>`;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    if (!id) return;
    const el = document.getElementById(id);
    if (el) el.remove();
}


// ── Staged Image Helpers ─────────────────────────────────────────────────────

function _removeImagePreview() {
    const preview = document.getElementById('image-preview-bar');
    if (preview) preview.remove();
}

function cancelStagedImage() {
    _stagedImage = null;
    _removeImagePreview();
    const chatInput = document.getElementById('chat-input');
    if (chatInput) chatInput.placeholder = "Ask Something";
}


// ── Retry Failed Question ────────────────────────────────────────────────────

function retryLastQuestion(btn) {
    if (!_lastQuestion) return;
    // Remove the failed message div
    const msgDiv = btn.closest('.flex.w-full');
    if (msgDiv) msgDiv.remove();
    // Re-submit the question
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.value = _lastQuestion;
        // Trigger send
        document.getElementById('send-btn')?.click();
    }
}


// ── Follow-up Suggestions Generator ─────────────────────────────────────────
// Generates 2-3 contextual follow-up questions based on the user's original
// question and the route taken. No API call — instant, client-side.

function _generateFollowUps(originalQuestion, route) {
    const q = originalQuestion.toLowerCase();
    const suggestions = [];

    // Detect language
    const isArabic = /[\u0600-\u06FF]/.test(originalQuestion);
    const isFrench = /\b(le|la|les|des|est|que|qui|quel|dans|pour)\b/i.test(originalQuestion);

    // Topic-based suggestions
    if (/taux|inflation|مع[دا]ل|تضخم/i.test(q)) {
        if (isFrench) {
            suggestions.push("Quelle est l'évolution de l'inflation sur les 3 dernières années ?");
            suggestions.push("Quel impact sur le pouvoir d'achat ?");
        } else if (isArabic) {
            suggestions.push("ما هو تأثير التضخم على القدرة الشرائية؟");
            suggestions.push("كيف تطور معدل التضخم خلال السنوات الأخيرة؟");
        } else {
            suggestions.push("How has inflation evolved over the past 3 years?");
            suggestions.push("What is the impact on purchasing power?");
        }
    } else if (/bourse|masi|madex|بورصة|سوق/i.test(q)) {
        if (isFrench) {
            suggestions.push("Quels sont les secteurs les plus performants ?");
            suggestions.push("Quelle est la capitalisation boursière totale ?");
        } else if (isArabic) {
            suggestions.push("ما هي القطاعات الأكثر أداءً؟");
            suggestions.push("ما هي القيمة السوقية الإجمالية؟");
        } else {
            suggestions.push("Which sectors are performing best?");
            suggestions.push("What is the total market capitalization?");
        }
    } else if (/dette|budget|ميزانية|دين/i.test(q)) {
        if (isFrench) {
            suggestions.push("Quel est le ratio dette/PIB actuel ?");
            suggestions.push("Comment évolue le déficit budgétaire ?");
        } else if (isArabic) {
            suggestions.push("ما هي نسبة الدين إلى الناتج المحلي؟");
            suggestions.push("كيف يتطور العجز في الميزانية؟");
        } else {
            suggestions.push("What is the current debt-to-GDP ratio?");
            suggestions.push("How is the budget deficit evolving?");
        }
    } else if (/bank|bkam|بنك|المغرب/i.test(q)) {
        if (isFrench) {
            suggestions.push("Quelle est la politique monétaire actuelle ?");
            suggestions.push("Comment évolue la liquidité bancaire ?");
        } else if (isArabic) {
            suggestions.push("ما هي السياسة النقدية الحالية؟");
            suggestions.push("كيف تتطور السيولة البنكية؟");
        } else {
            suggestions.push("What is the current monetary policy stance?");
            suggestions.push("How is banking liquidity evolving?");
        }
    } else {
        // Generic follow-ups based on route
        if (route && route.includes("SQL")) {
            suggestions.push(isFrench ? "Montrer les données sur 30 jours" : isArabic ? "عرض البيانات على 30 يوما" : "Show data for the last 30 days");
        }
        if (route && route.includes("Vector")) {
            suggestions.push(isFrench ? "Pouvez-vous détailler davantage ?" : isArabic ? "هل يمكنك التفصيل أكثر؟" : "Can you elaborate on this?");
        }
        if (suggestions.length === 0) {
            if (isFrench) suggestions.push("Pouvez-vous détailler davantage ?");
            else if (isArabic) suggestions.push("هل يمكنك التفصيل أكثر؟");
            else suggestions.push("Can you elaborate further?");
        }
    }

    return suggestions.slice(0, 3);
}


// ── Answer Feedback ──────────────────────────────────────────────────────────
// Sends thumbs up/down feedback to the backend for quality tracking.
// Falls back silently if the endpoint doesn't exist yet.

function _sendFeedback(question, answer, rating) {
    try {
        authFetch(`${API_BASE}/query/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question.substring(0, 500),
                answer: answer.substring(0, 1000),
                rating: rating, // "positive" or "negative"
                timestamp: new Date().toISOString(),
            }),
        }).catch(() => {}); // fire-and-forget
    } catch (e) { /* non-critical */ }
}


// ── Dark Mode Toggle ─────────────────────────────────────────────────────────

function toggleDarkMode() {
    const html = document.documentElement;
    const isDark = html.classList.toggle('dark');
    localStorage.setItem('dark_mode', isDark.toString());
    // Update icon
    const icon = document.getElementById('dark-mode-icon');
    if (icon) {
        icon.className = isDark ? 'fa-solid fa-sun text-[1.1rem]' : 'fa-solid fa-moon text-[1.1rem]';
    }
}

// Set icon state on load
document.addEventListener('DOMContentLoaded', () => {
    const isDark = document.documentElement.classList.contains('dark');
    const icon = document.getElementById('dark-mode-icon');
    if (icon && isDark) {
        icon.className = 'fa-solid fa-sun text-[1.1rem]';
    }
});


// ── Source Card Extraction ───────────────────────────────────────────────────
// Parses the context string to extract document references for clickable cards.
// Matches the HybridMerger output, which produces lines like:
//   **Excerpt 1** — [filename.pdf | page: 3 | score: 0.5480](pdf-viewer:...&page=3)
//   **Excerpt 1** — `[SOURCE: filename.pdf | page: 3 | score: 0.5480]`
// And a header table row:
//   | 1 | filename.pdf | Pg 3 | 0.5480 |

function _extractSourceCards(context) {
    const cards = [];
    const seen = new Set();

    // Pattern 1: Excerpt with markdown link  → [filename | page: N | score: 0.XXXX](pdf-viewer:...)
    const linkRe = /\[([^\]|]+?)(?:\s*\|\s*page:\s*(\d+))?(?:\s*\|\s*[^\]]*?score:\s*([\d.]+))?\s*\]\(pdf-viewer:/g;
    // Pattern 2: Excerpt with [SOURCE: ...] fallback
    const srcRe  = /\[SOURCE:\s*([^|\]]+?)(?:\s*\|\s*page:\s*(\d+))?(?:\s*\|\s*[^\]]*?score:\s*([\d.]+))?\s*\]/g;
    // Pattern 3: Old summary table row | N | filename | Pg X | 0.XXXX |
    const tableRe = /\|\s*\d+\s*\|\s*([^|]+?\.(?:pdf|md|html?|txt))\s*\|\s*([^|]*?)\s*\|\s*([\d.]+)\s*\|/gi;

    const pushCard = (file, page, score) => {
        if (!file) return;
        file = file.trim();
        const key = `${file}-${page || ''}`;
        if (seen.has(key)) return;
        seen.add(key);
        const shortName = file.split(/[/\\]/).pop().replace(/^(ammc_|hcp_|bkam_|cdc_)/, '').substring(0, 35);
        cards.push({ file, page: page || "", score: score || "", shortName });
    };

    let m;
    while ((m = linkRe.exec(context)) !== null)  pushCard(m[1], m[2], m[3]);
    while ((m = srcRe.exec(context))  !== null)  pushCard(m[1], m[2], m[3]);
    while ((m = tableRe.exec(context)) !== null) {
        // Pattern 3 page is in the metadata column like "Pg 3, 2022" or "OCR/Vision"
        const meta = m[2] || "";
        const pageMatch = meta.match(/Pg\s*(\d+)/i);
        pushCard(m[1], pageMatch ? pageMatch[1] : "", m[3]);
    }

    return cards.slice(0, 5); // Max 5 source cards
}
