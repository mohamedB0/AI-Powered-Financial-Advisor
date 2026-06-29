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
                <p class="text-sm text-gray-500 mt-1">Connectez-vous pour continuer</p>
            </div>

            <div id="auth-error" class="hidden bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-2 mb-4"></div>

            <!-- Tab Switch -->
            <div class="flex mb-5 bg-gray-100 rounded-lg p-1">
                <button id="tab-login" class="flex-1 py-2 text-sm font-semibold rounded-md bg-white shadow text-gray-800 transition-all" onclick="switchAuthTab('login')">Connexion</button>
                <button id="tab-register" class="flex-1 py-2 text-sm font-medium rounded-md text-gray-500 transition-all" onclick="switchAuthTab('register')">Inscription</button>
            </div>

            <!-- Login Form -->
            <form id="login-form" onsubmit="handleLogin(event)">
                <div class="mb-4">
                    <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
                    <input type="email" id="login-email" required
                        class="w-full px-4 py-2.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                        placeholder="votre@email.com">
                </div>
                <div class="mb-5">
                    <label class="block text-sm font-medium text-gray-700 mb-1">Mot de passe</label>
                    <input type="password" id="login-password" required minlength="8"
                        class="w-full px-4 py-2.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                        placeholder="••••••••">
                </div>
                <button type="submit" id="login-btn"
                    class="w-full bg-[#1a2e1f] hover:bg-[#243b28] text-white font-semibold py-2.5 rounded-lg transition-colors">
                    Se connecter
                </button>
            </form>

            <!-- Register Form (hidden) -->
            <form id="register-form" class="hidden" onsubmit="handleRegister(event)">
                <div class="mb-3">
                    <label class="block text-sm font-medium text-gray-700 mb-1">Nom complet</label>
                    <input type="text" id="register-name"
                        class="w-full px-4 py-2.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                        placeholder="Votre nom">
                </div>
                <div class="mb-3">
                    <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
                    <input type="email" id="register-email" required
                        class="w-full px-4 py-2.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                        placeholder="votre@email.com">
                </div>
                <div class="mb-5">
                    <label class="block text-sm font-medium text-gray-700 mb-1">Mot de passe</label>
                    <input type="password" id="register-password" required minlength="8"
                        class="w-full px-4 py-2.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                        placeholder="Min. 8 caractères">
                </div>
                <button type="submit" id="register-btn"
                    class="w-full bg-[#1a2e1f] hover:bg-[#243b28] text-white font-semibold py-2.5 rounded-lg transition-colors">
                    Créer un compte
                </button>
            </form>

            <!-- Divider -->
            <div class="flex items-center my-5">
                <div class="flex-1 border-t border-gray-200"></div>
                <span class="px-3 text-xs text-gray-400">ou</span>
                <div class="flex-1 border-t border-gray-200"></div>
            </div>

            <!-- Google Sign-In -->
            <div id="google-btn-container" class="flex justify-center">
                <!-- Google button rendered here -->
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // Render Google Sign-In button
    _renderGoogleButton();
}

function switchAuthTab(tab) {
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const tabLogin = document.getElementById('tab-login');
    const tabRegister = document.getElementById('tab-register');
    const errorEl = document.getElementById('auth-error');

    if (errorEl) errorEl.classList.add('hidden');

    if (tab === 'login') {
        loginForm.classList.remove('hidden');
        registerForm.classList.add('hidden');
        tabLogin.className = 'flex-1 py-2 text-sm font-semibold rounded-md bg-white shadow text-gray-800 transition-all';
        tabRegister.className = 'flex-1 py-2 text-sm font-medium rounded-md text-gray-500 transition-all';
    } else {
        loginForm.classList.add('hidden');
        registerForm.classList.remove('hidden');
        tabRegister.className = 'flex-1 py-2 text-sm font-semibold rounded-md bg-white shadow text-gray-800 transition-all';
        tabLogin.className = 'flex-1 py-2 text-sm font-medium rounded-md text-gray-500 transition-all';
    }
}

async function handleLogin(e) {
    e.preventDefault();
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    const errorEl = document.getElementById('auth-error');
    const btn = document.getElementById('login-btn');

    btn.disabled = true;
    btn.innerText = 'Connexion...';
    errorEl.classList.add('hidden');

    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Login failed');

        setToken(data.access_token);
        setUser(data.user);
        document.getElementById('auth-modal').remove();
        showToast(`Welcome, ${data.user.full_name || data.user.email.split('@')[0]}!`, 'success', 3000);
        updateAuthUI();
    } catch (err) {
        errorEl.innerText = err.message;
        errorEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.innerText = 'Se connecter';
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const name = document.getElementById('register-name').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    const errorEl = document.getElementById('auth-error');
    const btn = document.getElementById('register-btn');

    btn.disabled = true;
    btn.innerText = 'Création...';
    errorEl.classList.add('hidden');

    try {
        const res = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, full_name: name }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Registration failed');

        setToken(data.access_token);
        setUser(data.user);
        document.getElementById('auth-modal').remove();
        showToast(`Account created! Welcome, ${data.user.full_name || data.user.email.split('@')[0]}!`, 'success', 3000);
        updateAuthUI();
    } catch (err) {
        errorEl.innerText = err.message;
        errorEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.innerText = 'Créer un compte';
    }
}

function _renderGoogleButton(_attempt = 0) {
    const container = document.getElementById('google-btn-container');
    if (!container) return;  // modal closed

    // Missing client ID is a hard configuration error — tell the user instead
    // of retrying forever (the old behaviour, which silently showed nothing).
    if (!window.GOOGLE_CLIENT_ID) {
        console.error('[Auth] GOOGLE_CLIENT_ID is empty. Set it in .env and restart Flask.');
        _renderGoogleFallback(container, 'Google Sign-In is not configured.');
        return;
    }

    // Wait for the GSI SDK to load, but only for a bounded time (~6s).
    if (!window.google || !window.google.accounts || !window.google.accounts.id) {
        if (_attempt < 30) {
            setTimeout(() => _renderGoogleButton(_attempt + 1), 200);
        } else {
            console.error('[Auth] Google Identity SDK failed to load (network/blocker?).');
            _renderGoogleFallback(container, 'Could not load Google Sign-In.');
        }
        return;
    }

    try {
        google.accounts.id.initialize({
            client_id: window.GOOGLE_CLIENT_ID,
            callback: handleGoogleCredentialResponse,
            auto_select: false,
            cancel_on_tap_outside: false,
        });

        container.innerHTML = '';  // clear any previous fallback
        google.accounts.id.renderButton(container, {
            theme: 'outline',
            size: 'large',
            width: 320,
            text: 'signin_with',
            shape: 'rectangular',
            logo_alignment: 'center',
        });

        // GSI renders asynchronously and stays EMPTY (no error thrown) when the
        // current origin is not an Authorized JavaScript origin for this client.
        // Detect that empty render and show a fallback + a clear hint.
        setTimeout(() => {
            if (container && container.childElementCount === 0) {
                console.error(
                    `[Auth] Google button rendered empty. Most likely "${window.location.origin}" ` +
                    `is not in the client's Authorized JavaScript origins ` +
                    `(Google Cloud Console → Credentials → OAuth client).`
                );
                _renderGoogleFallback(
                    container,
                    `This address (${window.location.origin}) is not authorized for Google Sign-In.`
                );
            }
        }, 1200);
    } catch (err) {
        console.error('[Auth] Google button init failed:', err);
        _renderGoogleFallback(container, 'Google Sign-In failed to initialize.');
    }
}

/**
 * Fallback shown when the official Google button can't render.
 * Lets the user trigger the Google One-Tap/prompt flow manually, and surfaces
 * the underlying reason so login problems are diagnosable instead of silent.
 */
function _renderGoogleFallback(container, reason) {
    if (!container) return;
    container.innerHTML = `
        <div class="w-full">
            <button type="button" id="google-fallback-btn"
                class="w-full flex items-center justify-center gap-2.5 border border-gray-300 rounded-lg py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors">
                <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
                    <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.71-1.57 2.68-3.89 2.68-6.62z"/>
                    <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.81.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18z"/>
                    <path fill="#FBBC05" d="M3.97 10.72a5.4 5.4 0 0 1 0-3.44V4.95H.96a9 9 0 0 0 0 8.1l3.01-2.33z"/>
                    <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58A9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z"/>
                </svg>
                Continuer avec Google
            </button>
            ${reason ? `<p class="text-[11px] text-gray-400 text-center mt-2">${reason}</p>` : ''}
        </div>
    `;
    const btn = document.getElementById('google-fallback-btn');
    if (btn) {
        btn.addEventListener('click', () => {
            if (window.google && google.accounts && google.accounts.id) {
                try {
                    google.accounts.id.initialize({
                        client_id: window.GOOGLE_CLIENT_ID,
                        callback: handleGoogleCredentialResponse,
                    });
                    google.accounts.id.prompt();
                } catch (e) {
                    showToast('Google Sign-In unavailable. Use email/password.', 'warning');
                }
            } else {
                showToast('Google Sign-In unavailable. Use email/password.', 'warning');
            }
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

    // ── Token expiry watcher ─────────────────────────────────────────────
    // Check every 5 minutes if the token is about to expire. Warn the user
    // 30 minutes before expiry so they can re-login without losing work.
    _startTokenExpiryWatcher();
});

function _startTokenExpiryWatcher() {
    setInterval(() => {
        const token = getToken();
        if (!token) return;
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            const expiresAt = payload.exp * 1000;
            const now = Date.now();
            const remaining = expiresAt - now;

            // Warn 30 minutes before expiry
            if (remaining > 0 && remaining < 30 * 60 * 1000) {
                const mins = Math.round(remaining / 60000);
                showToast(`Session expires in ${mins} min. Save your work and re-login.`, 'warning', 8000);
            }
            // Already expired
            if (remaining <= 0) {
                clearAuth();
                updateAuthUI();
                showToast('Session expired. Please sign in again.', 'warning');
                showLoginModal();
            }
        } catch (e) { /* malformed token, ignore */ }
    }, 5 * 60 * 1000); // Check every 5 minutes
}
