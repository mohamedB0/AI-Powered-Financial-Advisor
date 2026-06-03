/**
 * auth.js — Frontend authentication module (Google OAuth)
 * Handles Google Sign-In, token storage, and auth-aware API calls.
 */

// API base URL (defined here so it's available before main.js loads)
// In development: Flask on :5000, FastAPI on :8000 → use explicit port
// In production behind Nginx: both on same domain → no port needed
const API_BASE = (() => {
    if (window.location.protocol === "file:") return "http://127.0.0.1:8000";
    const port = window.location.port;
    // If we're on port 5000 (dev Flask), point to :8000 (dev FastAPI)
    if (port === "5000") {
        const host = window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname;
        return window.location.protocol + "//" + host + ":8000";
    }
    // Production: API is proxied on the same origin (Nginx handles routing)
    return window.location.origin;
})();

const AUTH_TOKEN_KEY = 'fin_advisor_token';
const AUTH_USER_KEY = 'fin_advisor_user';

// ── Shared Utilities ─────────────────────────────────────────────────────────

/**
 * Format a Date into a human-friendly message timestamp.
 * - Just now / Xm ago / Xh ago for today
 * - "Yesterday 14:30" for yesterday
 * - "12 Jan 14:30" for older
 */
function _formatMsgTime(date) {
    const now = new Date();
    const diffMs = now - date;
    const diffMin = Math.floor(diffMs / 60000);
    const diffHr = Math.floor(diffMs / 3600000);

    if (diffMin < 1) return 'Just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHr < 24 && date.getDate() === now.getDate()) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (date.getDate() === yesterday.getDate() && date.getMonth() === yesterday.getMonth()) {
        return `Yesterday ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }

    return date.toLocaleDateString([], { day: 'numeric', month: 'short' }) +
        ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * Show a toast notification.
 * @param {string} message - Text to display
 * @param {string} type - 'success' | 'error' | 'warning' | 'info'
 * @param {number} duration - Auto-dismiss in ms (default 4000, 0 = manual dismiss)
 */
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const icons = {
        success: 'fa-solid fa-circle-check',
        error: 'fa-solid fa-circle-exclamation',
        warning: 'fa-solid fa-triangle-exclamation',
        info: 'fa-solid fa-circle-info',
    };

    const colors = {
        success: { bg: 'bg-emerald-50', border: 'border-emerald-200', icon: 'text-emerald-500', text: 'text-emerald-800' },
        error: { bg: 'bg-red-50', border: 'border-red-200', icon: 'text-red-500', text: 'text-red-800' },
        warning: { bg: 'bg-amber-50', border: 'border-amber-200', icon: 'text-amber-500', text: 'text-amber-800' },
        info: { bg: 'bg-blue-50', border: 'border-blue-200', icon: 'text-blue-500', text: 'text-blue-800' },
    };

    const c = colors[type] || colors.info;
    const icon = icons[type] || icons.info;

    const toast = document.createElement('div');
    toast.className = `toast-item pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-xl border shadow-lg ${c.bg} ${c.border} max-w-sm w-full`;
    toast.innerHTML = `
        <i class="${icon} ${c.icon} text-base mt-0.5 flex-shrink-0"></i>
        <p class="text-[13px] ${c.text} leading-snug flex-1">${message}</p>
        <button onclick="dismissToast(this.parentElement)" class="flex-shrink-0 w-5 h-5 flex items-center justify-center rounded text-gray-400 hover:text-gray-600 transition-colors">
            <i class="fa-solid fa-xmark text-[11px]"></i>
        </button>
    `;

    container.appendChild(toast);

    // Trigger entrance animation
    requestAnimationFrame(() => toast.classList.add('toast-enter'));

    // Auto-dismiss
    if (duration > 0) {
        setTimeout(() => dismissToast(toast), duration);
    }

    return toast;
}

function dismissToast(el) {
    if (!el || el.classList.contains('toast-exit')) return;
    el.classList.add('toast-exit');
    el.addEventListener('animationend', () => el.remove(), { once: true });
}

// ── Help Modal ────────────────────────────────────────────────────────────────
function showHelpModal() {
    const modal = document.getElementById('help-modal');
    if (modal) modal.classList.remove('hidden');
}

function closeHelpModal() {
    const modal = document.getElementById('help-modal');
    if (modal) modal.classList.add('hidden');
}

// Close modals with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeHelpModal();
        const settingsModal = document.getElementById('settings-modal');
        if (settingsModal) settingsModal.classList.add('hidden');
    }
});

// ── Token Management ─────────────────────────────────────────────────────────

function getToken() {
    return localStorage.getItem(AUTH_TOKEN_KEY);
}

function setToken(token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
}

function getUser() {
    const raw = localStorage.getItem(AUTH_USER_KEY);
    return raw ? JSON.parse(raw) : null;
}

function setUser(user) {
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
}

function clearAuth() {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
}

function isAuthenticated() {
    const token = getToken();
    if (!token) return false;
    try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        return payload.exp * 1000 > Date.now();
    } catch {
        return false;
    }
}

// ── Auth-aware fetch wrapper ─────────────────────────────────────────────────

/**
 * Wrapper around fetch() that automatically injects the Authorization header.
 * If the response is 401, redirects to login.
 */
async function authFetch(url, options = {}) {
    const token = getToken();
    if (!options.headers) {
        options.headers = {};
    }
    if (token) {
        options.headers['Authorization'] = `Bearer ${token}`;
    }
    const response = await fetch(url, options);
    if (response.status === 401) {
        clearAuth();
        showToast('Session expired. Please sign in again.', 'warning');
        showLoginModal();
        throw new Error('Authentication required');
    }
    return response;
}

// ── Google Sign-In Callback ──────────────────────────────────────────────────

/**
 * Called by Google Sign-In when user completes authentication.
 * Sends the ID token to our backend for verification.
 */
async function handleGoogleCredentialResponse(response) {
    const errorEl = document.getElementById('auth-error');
    const btnContainer = document.getElementById('google-btn-container');

    if (errorEl) errorEl.classList.add('hidden');

    const url = `/api/auth/google`;  // Same-origin proxy through Flask
    console.log('[Auth] Sending Google token to:', url);

    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id_token: response.credential }),
        });
        console.log('[Auth] Response status:', res.status);
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || 'Authentication failed');
        }

        setToken(data.access_token);
        setUser(data.user);

        // Remove the modal
        const modal = document.getElementById('auth-modal');
        if (modal) modal.remove();

        showToast(`Welcome, ${data.user.full_name || data.user.email.split('@')[0]}!`, 'success', 3000);
        updateAuthUI();
    } catch (err) {
        console.error('[Auth] Error details:', err);
        if (errorEl) {
            // Distinguish network/CORS errors from backend errors
            if (err instanceof TypeError && err.message === 'Failed to fetch') {
                errorEl.innerHTML = `
                    <strong>Impossible de contacter le serveur API</strong><br>
                    <span style="font-size:12px">Vérifiez que le backend FastAPI tourne sur <code>${API_BASE}</code></span>
                `;
            } else {
                errorEl.innerText = err.message;
            }
            errorEl.classList.remove('hidden');
        }
    }
}

function logout() {
    clearAuth();
    showToast('You have been signed out.', 'info', 3000);
    // Revoke Google session if available
    if (window.google && google.accounts) {
        google.accounts.id.disableAutoSelect();
    }
    setTimeout(() => window.location.reload(), 500);
}

// ── Login Modal UI ───────────────────────────────────────────────────────────

function showLoginModal() {
    // Remove existing modal if any
    const existing = document.getElementById('auth-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'auth-modal';
    modal.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm';
    modal.innerHTML = `
        <div class="bg-white rounded-2xl shadow-2xl w-full max-w-md p-8 mx-4">
            <div class="text-center mb-6">
                <h2 class="text-2xl font-bold text-[#1a2e1f]" style="font-family: 'Playfair Display', serif;">
                    Alpha <span class="text-emerald-600 italic">Insight</span>
                </h2>
                <p class="text-sm text-gray-500 mt-2">Connectez-vous avec votre compte Google</p>
            </div>

            <div id="auth-error" class="hidden bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-2 mb-4"></div>

            <!-- Google Sign-In Button Container -->
            <div id="google-btn-container" class="flex justify-center py-4">
                <!-- Google button will be rendered here -->
            </div>

            <p class="text-xs text-gray-400 text-center mt-4">
                En vous connectant, vous acceptez nos conditions d'utilisation.
            </p>
        </div>
    `;
    document.body.appendChild(modal);

    // Render Google Sign-In button
    _renderGoogleButton();
}

function _renderGoogleButton() {
    if (!window.google || !window.GOOGLE_CLIENT_ID) {
        // Google SDK not loaded yet, retry after a short delay
        setTimeout(_renderGoogleButton, 200);
        return;
    }

    google.accounts.id.initialize({
        client_id: window.GOOGLE_CLIENT_ID,
        callback: handleGoogleCredentialResponse,
        auto_select: false,
        cancel_on_tap_outside: false,
    });

    const container = document.getElementById('google-btn-container');
    if (container) {
        google.accounts.id.renderButton(container, {
            theme: 'outline',
            size: 'large',
            width: 320,
            text: 'signin_with',
            shape: 'rectangular',
            logo_alignment: 'center',
        });
    }
}

// ── Auth UI state ────────────────────────────────────────────────────────────

function _getInitials(user) {
    if (!user) return '?';
    if (user.full_name && user.full_name.trim()) {
        const parts = user.full_name.trim().split(/\s+/);
        if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
        return parts[0][0].toUpperCase();
    }
    return user.email ? user.email[0].toUpperCase() : '?';
}

function toggleUserMenu() {
    const menu = document.getElementById('user-menu');
    if (!menu) return;
    menu.classList.toggle('hidden');
}

// Close user menu when clicking outside
document.addEventListener('click', (e) => {
    const menu = document.getElementById('user-menu');
    const btn = document.getElementById('user-avatar-btn');
    if (!menu || !btn) return;
    if (!btn.contains(e.target) && !menu.contains(e.target)) {
        menu.classList.add('hidden');
    }
});

function updateAuthUI() {
    const user = getUser();
    const initialsEl = document.getElementById('user-initials');
    const avatarBtn = document.getElementById('user-avatar-btn');
    const nameEl = document.getElementById('menu-user-name');
    const emailEl = document.getElementById('menu-user-email');
    const loginBtn = document.getElementById('menu-login-btn');
    const logoutBtn = document.getElementById('menu-logout-btn');

    if (!avatarBtn) return;

    if (user && isAuthenticated()) {
        // Logged in state
        const initials = _getInitials(user);

        // If user has a Google profile picture, show it instead of initials
        if (user.picture) {
            avatarBtn.innerHTML = `<img src="${user.picture}" alt="Avatar" class="w-full h-full rounded-full object-cover">`;
        } else {
            avatarBtn.innerHTML = `<span id="user-initials" class="text-white font-bold text-[13px] leading-none select-none">${initials}</span>`;
        }

        avatarBtn.className = 'w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-emerald-700 shadow-lg border-2 border-emerald-400/40 flex items-center justify-center cursor-pointer hover:scale-110 hover:border-emerald-300 transition-all duration-200 overflow-hidden';
        if (nameEl) nameEl.textContent = user.full_name || user.email.split('@')[0];
        if (emailEl) emailEl.textContent = user.email || '';
        if (loginBtn) loginBtn.style.display = 'none';
        if (logoutBtn) logoutBtn.style.display = 'flex';
    } else {
        // Logged out state
        avatarBtn.innerHTML = `<span id="user-initials" class="text-white font-bold text-[13px] leading-none select-none">?</span>`;
        avatarBtn.className = 'w-10 h-10 rounded-full bg-gradient-to-br from-gray-400 to-gray-600 shadow-lg border-2 border-gray-300/40 flex items-center justify-center cursor-pointer hover:scale-110 hover:border-gray-200 transition-all duration-200';
        if (nameEl) nameEl.textContent = 'Non connecté';
        if (emailEl) emailEl.textContent = 'Cliquez pour vous connecter';
        if (loginBtn) loginBtn.style.display = 'flex';
        if (logoutBtn) logoutBtn.style.display = 'none';
    }
}

// ── Initialization ───────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    updateAuthUI();

    if (!isAuthenticated()) {
        showLoginModal();
        return;
    }

    // Verify token is still valid server-side
    try {
        const res = await fetch(`${API_BASE}/auth/me`, {
            headers: { 'Authorization': `Bearer ${getToken()}` }
        });
        if (res.status === 401) {
            clearAuth();
            updateAuthUI();
            showLoginModal();
        }
    } catch (e) {
        // API not reachable — don't force logout
    }
});
