/**
 * history.js — Chat history sidebar module
 * Manages conversations list, auto-save, and conversation switching.
 */

let _currentConversationId = null;
let _conversations = [];

// ── Sidebar Toggle ───────────────────────────────────────────────────────────

function toggleHistorySidebar() {
    const sidebar = document.getElementById('history-sidebar');
    const showBtn = document.getElementById('show-history-btn');

    if (sidebar.classList.contains('hidden')) {
        // Show sidebar
        sidebar.classList.remove('hidden');
        showBtn.classList.add('hidden');
        localStorage.setItem('history_sidebar_visible', 'true');
    } else {
        // Hide sidebar
        sidebar.classList.add('hidden');
        showBtn.classList.remove('hidden');
        showBtn.classList.add('flex');
        localStorage.setItem('history_sidebar_visible', 'false');
    }
}

function toggleMobileHistory() {
    const sidebar = document.getElementById('history-sidebar');
    const isOpen = sidebar.classList.contains('mobile-open');

    if (isOpen) {
        closeMobileHistory();
    } else {
        sidebar.classList.add('mobile-open');
        // Add backdrop
        const backdrop = document.createElement('div');
        backdrop.id = 'mobile-history-backdrop';
        backdrop.onclick = closeMobileHistory;
        document.body.appendChild(backdrop);
    }
}

function closeMobileHistory() {
    const sidebar = document.getElementById('history-sidebar');
    sidebar.classList.remove('mobile-open');
    const backdrop = document.getElementById('mobile-history-backdrop');
    if (backdrop) backdrop.remove();
}

// Restore sidebar state on load
document.addEventListener('DOMContentLoaded', () => {
    const visible = localStorage.getItem('history_sidebar_visible');
    if (visible === 'false') {
        const sidebar = document.getElementById('history-sidebar');
        const showBtn = document.getElementById('show-history-btn');
        if (sidebar && showBtn) {
            sidebar.classList.add('hidden');
            showBtn.classList.remove('hidden');
            showBtn.classList.add('flex');
        }
    }
});

// ── API Calls ────────────────────────────────────────────────────────────────

async function fetchConversations() {
    try {
        const res = await authFetch(`${API_BASE}/conversations/`);
        if (!res.ok) return [];
        return await res.json();
    } catch {
        return [];
    }
}

async function createConversation(title) {
    try {
        const res = await authFetch(`${API_BASE}/conversations/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: title || 'New Chat' }),
        });
        if (!res.ok) return null;
        return await res.json();
    } catch {
        return null;
    }
}

async function fetchConversationMessages(convId) {
    try {
        const res = await authFetch(`${API_BASE}/conversations/${convId}`);
        if (!res.ok) return null;
        return await res.json();
    } catch {
        return null;
    }
}

async function saveMessage(convId, role, content, route) {
    if (!convId || !content) return;
    try {
        await authFetch(`${API_BASE}/conversations/${convId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ role, content, route: route || null }),
        });
    } catch (e) {
        console.error('Failed to save message:', e);
    }
}

async function deleteConversation(convId) {
    try {
        await authFetch(`${API_BASE}/conversations/${convId}`, { method: 'DELETE' });
    } catch (e) {
        console.error('Failed to delete conversation:', e);
    }
}

async function renameConversation(convId, newTitle) {
    try {
        await authFetch(`${API_BASE}/conversations/${convId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle }),
        });
    } catch (e) {
        console.error('Failed to rename conversation:', e);
    }
}

// ── UI Rendering ─────────────────────────────────────────────────────────────

function renderHistorySidebar(conversations) {
    const container = document.getElementById('history-list');
    if (!container) return;

    _conversations = conversations;

    if (!conversations.length) {
        container.innerHTML = `
            <div class="px-4 py-8 text-center text-[12px] text-gray-400">
                <i class="fa-regular fa-clock text-lg mb-2 block"></i>
                No conversations yet
            </div>
        `;
        return;
    }

    // Group by date
    const today = new Date().toDateString();
    const yesterday = new Date(Date.now() - 86400000).toDateString();

    let html = '';
    let lastGroup = '';

    conversations.forEach(conv => {
        const date = new Date(conv.updated_at || conv.created_at);
        const dateStr = date.toDateString();
        let group = '';
        if (dateStr === today) group = "Aujourd'hui";
        else if (dateStr === yesterday) group = 'Hier';
        else group = date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' });

        if (group !== lastGroup) {
            html += `<div class="px-4 pt-4 pb-1 text-[10px] font-semibold uppercase tracking-wider text-gray-400">${group}</div>`;
            lastGroup = group;
        }

        const isActive = conv.id === _currentConversationId;
        const activeCls = isActive
            ? 'bg-emerald-50 border-l-2 border-emerald-500 text-gray-900'
            : 'border-l-2 border-transparent hover:bg-gray-50 text-gray-600';

        const preview = conv.last_message
            ? conv.last_message.substring(0, 50) + (conv.last_message.length > 50 ? '...' : '')
            : '';

        html += `
            <div class="group relative ${activeCls} px-4 py-2.5 cursor-pointer transition-all duration-150"
                 onclick="loadConversation(${conv.id})" data-conv-id="${conv.id}">
                <div class="flex items-start justify-between gap-2">
                    <div class="min-w-0 flex-1">
                        <p class="text-[13px] font-medium truncate leading-tight">${escapeHtml(conv.title)}</p>
                        ${preview ? `<p class="text-[11px] text-gray-400 truncate mt-0.5">${escapeHtml(preview)}</p>` : ''}
                    </div>
                    <button onclick="event.stopPropagation(); showConvMenu(${conv.id}, event)"
                        class="opacity-0 group-hover:opacity-100 shrink-0 w-6 h-6 flex items-center justify-center rounded text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-all">
                        <i class="fa-solid fa-ellipsis text-[11px]"></i>
                    </button>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Conversation Context Menu ────────────────────────────────────────────────

function showConvMenu(convId, event) {
    // Remove existing menu
    const existing = document.getElementById('conv-context-menu');
    if (existing) existing.remove();

    const menu = document.createElement('div');
    menu.id = 'conv-context-menu';
    menu.className = 'fixed bg-white rounded-lg shadow-xl border border-gray-100 py-1.5 z-[200] w-40';
    menu.style.left = event.clientX + 'px';
    menu.style.top = event.clientY + 'px';

    menu.innerHTML = `
        <button onclick="promptRenameConv(${convId})" class="w-full text-left px-3 py-1.5 text-[12px] text-gray-600 hover:bg-gray-50 flex items-center gap-2">
            <i class="fa-solid fa-pen text-[10px] w-4 text-center text-gray-400"></i> Rename
        </button>
        <button onclick="confirmDeleteConv(${convId})" class="w-full text-left px-3 py-1.5 text-[12px] text-red-600 hover:bg-red-50 flex items-center gap-2">
            <i class="fa-solid fa-trash text-[10px] w-4 text-center"></i> Delete
        </button>
    `;

    document.body.appendChild(menu);

    // Close on click outside
    setTimeout(() => {
        document.addEventListener('click', function closeMenu() {
            menu.remove();
            document.removeEventListener('click', closeMenu);
        }, { once: true });
    }, 10);
}

function promptRenameConv(convId) {
    const conv = _conversations.find(c => c.id === convId);
    const newTitle = prompt('Rename conversation:', conv?.title || '');
    if (newTitle && newTitle.trim()) {
        renameConversation(convId, newTitle.trim()).then(() => refreshHistory());
    }
    document.getElementById('conv-context-menu')?.remove();
}

function confirmDeleteConv(convId) {
    document.getElementById('conv-context-menu')?.remove();
    if (confirm('Delete this conversation?')) {
        deleteConversation(convId).then(() => {
            if (_currentConversationId === convId) {
                _currentConversationId = null;
                resetChatUI();
            }
            refreshHistory();
        });
    }
}

// ── Conversation Loading ─────────────────────────────────────────────────────

async function loadConversation(convId) {
    const data = await fetchConversationMessages(convId);
    if (!data) return;

    _currentConversationId = convId;

    // Clear current chat and render messages
    const chatHistory = document.getElementById('chat-history');
    const emptyState = document.getElementById('empty-state');
    if (!chatHistory) return;

    chatHistory.innerHTML = '';

    if (data.messages.length > 0) {
        if (emptyState) emptyState.style.display = 'none';
        chatHistory.classList.remove('hidden');
        chatHistory.classList.add('flex');

        data.messages.forEach(msg => {
            addMessageToUI(msg.role, msg.content, msg.created_at || msg.timestamp);
        });

        chatHistory.scrollTop = chatHistory.scrollHeight;
    } else {
        resetChatUI();
    }

    // Update active state in sidebar
    document.querySelectorAll('[data-conv-id]').forEach(el => {
        if (parseInt(el.dataset.convId) === convId) {
            el.className = el.className.replace('border-transparent hover:bg-gray-50 text-gray-600', '');
            el.classList.add('bg-emerald-50', 'border-emerald-500', 'text-gray-900');
        } else {
            el.classList.remove('bg-emerald-50', 'border-emerald-500', 'text-gray-900');
            if (!el.className.includes('border-transparent')) {
                el.classList.add('border-transparent');
            }
        }
    });
}

function resetChatUI() {
    const chatHistory = document.getElementById('chat-history');
    const emptyState = document.getElementById('empty-state');
    if (chatHistory) {
        chatHistory.innerHTML = '';
        chatHistory.classList.add('hidden');
        chatHistory.classList.remove('flex');
    }
    if (emptyState) emptyState.style.display = '';
}

/**
 * Helper to add a message bubble to the chat UI.
 * Used by both loadConversation (history replay) and the live chat flow.
 */
function addMessageToUI(role, content, timestamp) {
    const chatHistory = document.getElementById('chat-history');
    if (!chatHistory) return;

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

    const bubbleRadius = role === 'user'
        ? 'rounded-[20px] rounded-br-[4px]'
        : 'rounded-[20px] rounded-bl-[4px]';

    const proseClass = role === 'user' ? 'prose-invert text-white' : 'prose-slate text-gray-700';

    const rendered = typeof marked !== 'undefined' ? marked.parse(content) : content;

    const timeStr = timestamp ? _formatMsgTime(new Date(timestamp)) : _formatMsgTime(new Date());
    const timeAlign = role === 'user' ? 'text-right' : 'text-left';

    msgDiv.innerHTML = `
        ${role === 'assistant' ? avatarStr : ''}
        <div class="relative max-w-[85%] flex flex-col">
            <div class="${bubbleRadius} px-5 py-4 ${bubbleColor}">
                <div class="msg-bubble prose prose-sm ${proseClass} max-w-none leading-[1.65]">${rendered}</div>
            </div>
            <span class="msg-timestamp ${timeAlign}">${timeStr}</span>
        </div>
        ${role === 'user' ? avatarStr : ''}
    `;

    chatHistory.appendChild(msgDiv);
}

// ── New Chat ─────────────────────────────────────────────────────────────────

async function startNewConversation() {
    _currentConversationId = null;
    resetChatUI();
    refreshHistory();
}

/**
 * Ensures a conversation exists for the current chat session.
 * Called when the first message is sent.
 */
async function ensureConversation() {
    if (_currentConversationId) return _currentConversationId;
    const conv = await createConversation('New Chat');
    if (conv) {
        _currentConversationId = conv.id;
        refreshHistory();
    }
    return _currentConversationId;
}

// ── Refresh ──────────────────────────────────────────────────────────────────

async function refreshHistory() {
    if (!isAuthenticated()) return;
    const conversations = await fetchConversations();
    renderHistorySidebar(conversations);
}

// ── Getters ──────────────────────────────────────────────────────────────────

function getCurrentConversationId() {
    return _currentConversationId;
}

// ── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    if (isAuthenticated()) {
        refreshHistory();
    }
});
