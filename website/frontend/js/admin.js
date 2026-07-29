/* =====================================================
   admin.js — Fazal-e-Rashid Vision Admin Portal Logic
   Handles: Supabase Auth, session protection, query fetching,
            status updates, deletion, filtering, and search.
   ===================================================== */

'use strict';

(function initAdminPortal() {
  const loginView     = document.getElementById('login-view');
  const dashboardView = document.getElementById('dashboard-view');
  const loginForm     = document.getElementById('admin-login-form');
  const loginError    = document.getElementById('login-error');
  const loginBtn      = document.getElementById('login-btn');
  const logoutBtn     = document.getElementById('logout-btn');
  const userEmailDisp = document.getElementById('user-email-display');

  const queriesTbody  = document.getElementById('queries-tbody');
  const queriesEmpty  = document.getElementById('queries-empty');
  const searchInput   = document.getElementById('search-input');
  const tabBtns       = document.querySelectorAll('.tab-btn');

  const statTotal    = document.getElementById('stat-total');
  const statNew      = document.getElementById('stat-new');
  const statReplied  = document.getElementById('stat-replied');
  const statArchived = document.getElementById('stat-archived');

  let allQueries = [];
  let currentFilter = 'all';
  let searchQuery = '';

  /* ── Get Supabase Client ───────────────────────────── */
  function getClient() {
    return window.supabaseClient;
  }

  /* ── Auth State & Session Check ─────────────────────── */
  function checkSession() {
    const client = getClient();
    if (!client) {
      showLoginError('Supabase client not initialized. Check js/supabase-config.js');
      showUnauthenticatedUI();
      return;
    }

    client.auth.getSession().then(({ data: { session }, error }) => {
      if (error || !session) {
        showUnauthenticatedUI();
      } else {
        showAuthenticatedUI(session.user);
      }
    }).catch(err => {
      console.error('Session check failed:', err);
      showUnauthenticatedUI();
    });
  }

  function showUnauthenticatedUI() {
    loginView.style.display = 'flex';
    dashboardView.style.display = 'none';
    allQueries = [];
  }

  function showAuthenticatedUI(user) {
    loginView.style.display = 'none';
    dashboardView.style.display = 'block';
    if (userEmailDisp) userEmailDisp.textContent = user?.email || 'Admin';
    fetchQueries();
  }

  /* ── Login Form Handler ─────────────────────────────── */
  if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      hideLoginError();

      const email = document.getElementById('login-email')?.value.trim();
      const password = document.getElementById('login-password')?.value;

      if (!email || !password) {
        showLoginError('Please enter both email and password.');
        return;
      }

      const client = getClient();
      if (!client) {
        showLoginError('Supabase credentials missing in js/supabase-config.js');
        return;
      }

      loginBtn.disabled = true;
      loginBtn.textContent = 'Authenticating…';

      try {
        const { data, error } = await client.auth.signInWithPassword({ email, password });
        loginBtn.disabled = false;
        loginBtn.textContent = 'Sign In';

        if (error) {
          showLoginError(error.message || 'Invalid login credentials.');
        } else if (data?.session) {
          loginForm.reset();
          showAuthenticatedUI(data.session.user);
        }
      } catch (err) {
        loginBtn.disabled = false;
        loginBtn.textContent = 'Sign In';
        showLoginError('Connection error during sign-in.');
        console.error('Login error:', err);
      }
    });
  }

  /* ── Logout Handler ────────────────────────────────── */
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      const client = getClient();
      if (client) {
        await client.auth.signOut();
      }
      showUnauthenticatedUI();
    });
  }

  function showLoginError(msg) {
    if (!loginError) return;
    loginError.textContent = msg;
    loginError.style.display = 'block';
  }

  function hideLoginError() {
    if (!loginError) return;
    loginError.style.display = 'none';
  }

  /* ── Fetch Queries from Supabase ───────────────────── */
  async function fetchQueries() {
    const client = getClient();
    if (!client) return;

    queriesTbody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align:center; padding: 2.5rem; color: var(--text-muted);">
          Loading inquiries from database…
        </td>
      </tr>
    `;

    try {
      const { data, error } = await client
        .from('contact_queries')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Fetch queries error:', error);
        queriesTbody.innerHTML = `
          <tr>
            <td colspan="5" style="text-align:center; padding: 2rem; color: var(--accent-red);">
              ⚠️ Database Access Denied: ${escapeHtml(error.message)}. Ensure SQL script has been run in Supabase.
            </td>
          </tr>
        `;
        return;
      }

      allQueries = data || [];
      updateStats();
      renderQueries();
    } catch (err) {
      console.error('Unexpected error fetching queries:', err);
    }
  }

  /* ── Update Statistics Counters ────────────────────── */
  function updateStats() {
    const total = allQueries.length;
    const countNew = allQueries.filter(q => (q.status || 'new') === 'new').length;
    const countReplied = allQueries.filter(q => q.status === 'replied').length;
    const countArchived = allQueries.filter(q => q.status === 'archived').length;

    if (statTotal) statTotal.textContent = total;
    if (statNew) statNew.textContent = countNew;
    if (statReplied) statReplied.textContent = countReplied;
    if (statArchived) statArchived.textContent = countArchived;
  }

  /* ── Render Queries Table ───────────────────────────── */
  function renderQueries() {
    const filtered = allQueries.filter(q => {
      const status = (q.status || 'new').toLowerCase();
      const matchesFilter = (currentFilter === 'all') || (status === currentFilter);
      
      const term = searchQuery.toLowerCase();
      const matchesSearch = !term || 
        (q.name && q.name.toLowerCase().includes(term)) ||
        (q.email && q.email.toLowerCase().includes(term)) ||
        (q.message && q.message.toLowerCase().includes(term));

      return matchesFilter && matchesSearch;
    });

    if (filtered.length === 0) {
      queriesTbody.innerHTML = '';
      queriesEmpty.style.display = 'block';
      return;
    }

    queriesEmpty.style.display = 'none';

    queriesTbody.innerHTML = filtered.map(q => {
      const formattedDate = formatDate(q.created_at);
      const status = (q.status || 'new').toLowerCase();
      const badgeClass = status === 'replied' ? 'badge--replied' : status === 'archived' ? 'badge--archived' : 'badge--new';

      return `
        <tr data-id="${q.id}">
          <td>
            <div class="query-date">${formattedDate}</div>
          </td>
          <td>
            <div class="user-info__name">${escapeHtml(q.name)}</div>
            <a href="mailto:${escapeHtml(q.email)}?subject=Re:%20Inquiry%20with%20Fazal-e-Rashid%20Vision" class="user-info__email">${escapeHtml(q.email)}</a>
          </td>
          <td>
            <div class="query-message">${escapeHtml(q.message)}</div>
          </td>
          <td>
            <span class="badge ${badgeClass}">${escapeHtml(status)}</span>
          </td>
          <td>
            <div class="actions-cell">
              <a href="mailto:${escapeHtml(q.email)}" class="btn-action" title="Send Email">Reply</a>
              ${status !== 'replied' ? `<button class="btn-action btn-mark-replied" onclick="window.updateStatus('${q.id}', 'replied')">Mark Replied</button>` : ''}
              ${status !== 'archived' ? `<button class="btn-action btn-mark-archive" onclick="window.updateStatus('${q.id}', 'archived')">Archive</button>` : ''}
              ${status === 'archived' || status === 'replied' ? `<button class="btn-action" onclick="window.updateStatus('${q.id}', 'new')">Mark New</button>` : ''}
              <button class="btn-action btn-action--delete" onclick="window.deleteQuery('${q.id}')">Delete</button>
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }

  /* ── Status Update Handler ─────────────────────────── */
  window.updateStatus = async function(id, newStatus) {
    const client = getClient();
    if (!client) return;

    try {
      const { error } = await client
        .from('contact_queries')
        .update({ status: newStatus })
        .eq('id', id);

      if (error) {
        alert('Failed to update status: ' + error.message);
      } else {
        const item = allQueries.find(q => q.id === id);
        if (item) item.status = newStatus;
        updateStats();
        renderQueries();
      }
    } catch (err) {
      console.error('Update status failed:', err);
    }
  };

  /* ── Delete Handler ────────────────────────────────── */
  window.deleteQuery = async function(id) {
    if (!confirm('Are you sure you want to permanently delete this inquiry?')) return;

    const client = getClient();
    if (!client) return;

    try {
      const { error } = await client
        .from('contact_queries')
        .delete()
        .eq('id', id);

      if (error) {
        alert('Failed to delete query: ' + error.message);
      } else {
        allQueries = allQueries.filter(q => q.id !== id);
        updateStats();
        renderQueries();
      }
    } catch (err) {
      console.error('Delete query failed:', err);
    }
  };

  /* ── Filter Tabs ───────────────────────────────────── */
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.getAttribute('data-filter') || 'all';
      renderQueries();
    });
  });

  /* ── Search Input ──────────────────────────────────── */
  if (searchInput) {
    searchInput.addEventListener('input', e => {
      searchQuery = e.target.value.trim();
      renderQueries();
    });
  }

  /* ── Helpers ───────────────────────────────────────── */
  function formatDate(isoStr) {
    if (!isoStr) return '';
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      return isoStr;
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  /* ── Startup Auth Listeners ────────────────────────── */
  const client = getClient();
  if (client) {
    client.auth.onAuthStateChange((event, session) => {
      if (session) {
        showAuthenticatedUI(session.user);
      } else {
        showUnauthenticatedUI();
      }
    });
  }

  // Initial check
  checkSession();
})();
