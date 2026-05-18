/**
 * auth.js — Frontend authentication module
 * Handles login/register, token storage, and auth-aware API calls.
 */

const AUTH_TOKEN_KEY = 'fin_advisor_token';
const AUTH_USER_KEY = 'fin_advisor_user';

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
    // Check expiration (JWT payload is base64url-encoded)
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
        showLoginModal();
        throw new Error('Authentication required');
    }
    return response;
}

// ── Login / Register API calls ───────────────────────────────────────────────

async function loginUser(email, password) {
    const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.detail || 'Login failed');
    }
    setToken(data.access_token);
    setUser(data.user);
    return data;
}

async function registerUser(email, password, fullName) {
    const res = await fetch(`${API_BASE}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, full_name: fullName }),
    });
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.detail || 'Registration failed');
    }
    setToken(data.access_token);
    setUser(data.user);
    return data;
}

function logout() {
    clearAuth();
    window.location.reload();
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
            <div class="flex mb-6 bg-gray-100 rounded-lg p-1">
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
                <div class="mb-6">
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
                <div class="mb-6">
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
        </div>
    `;
    document.body.appendChild(modal);
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
        await loginUser(email, password);
        document.getElementById('auth-modal').remove();
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
        await registerUser(email, password, name);
        document.getElementById('auth-modal').remove();
        updateAuthUI();
    } catch (err) {
        errorEl.innerText = err.message;
        errorEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.innerText = 'Créer un compte';
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
        if (initialsEl) initialsEl.textContent = initials;
        avatarBtn.className = 'w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-emerald-700 shadow-lg border-2 border-emerald-400/40 flex items-center justify-center cursor-pointer hover:scale-110 hover:border-emerald-300 transition-all duration-200';
        if (nameEl) nameEl.textContent = user.full_name || user.email.split('@')[0];
        if (emailEl) emailEl.textContent = user.email || '';
        if (loginBtn) loginBtn.style.display = 'none';
        if (logoutBtn) logoutBtn.style.display = 'flex';
    } else {
        // Logged out state
        if (initialsEl) initialsEl.textContent = '?';
        avatarBtn.className = 'w-10 h-10 rounded-full bg-gradient-to-br from-gray-400 to-gray-600 shadow-lg border-2 border-gray-300/40 flex items-center justify-center cursor-pointer hover:scale-110 hover:border-gray-200 transition-all duration-200';
        if (nameEl) nameEl.textContent = 'Non connecté';
        if (emailEl) emailEl.textContent = 'Cliquez pour vous connecter';
        if (loginBtn) loginBtn.style.display = 'flex';
        if (logoutBtn) logoutBtn.style.display = 'none';
    }
}

// ── Initialization ───────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    updateAuthUI();
    // If not authenticated, show login modal
    if (!isAuthenticated()) {
        showLoginModal();
    }
});
