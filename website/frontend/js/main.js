/* =====================================================
   main.js — Fazal-e-Rashid Vision
   Handles: nav scroll, mobile menu, scroll reveal,
            active link, contact form validation
   ===================================================== */

'use strict';

/* ── Helpers ─────────────────────────────────────────── */
const qs  = (sel, ctx = document) => ctx.querySelector(sel);
const qsa = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

/* ── Nav scroll class ────────────────────────────────── */
(function initNavScroll() {
  const nav = qs('.nav');
  if (!nav) return;
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 30);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

/* ── Mobile hamburger ────────────────────────────────── */
(function initMobileMenu() {
  const btn    = qs('#nav-hamburger');
  const menu   = qs('#nav-mobile');
  if (!btn || !menu) return;

  btn.addEventListener('click', () => {
    const isOpen = menu.classList.toggle('open');
    btn.classList.toggle('open', isOpen);
    btn.setAttribute('aria-expanded', String(isOpen));
    btn.setAttribute('aria-label', isOpen ? 'Close menu' : 'Open menu');
  });

  // close on link click
  qsa('a', menu).forEach(a =>
    a.addEventListener('click', () => {
      menu.classList.remove('open');
      btn.classList.remove('open');
      btn.setAttribute('aria-expanded', 'false');
    })
  );

  // close on outside click
  document.addEventListener('click', e => {
    if (!btn.contains(e.target) && !menu.contains(e.target)) {
      menu.classList.remove('open');
      btn.classList.remove('open');
      btn.setAttribute('aria-expanded', 'false');
    }
  });
})();

/* ── Active nav link ─────────────────────────────────── */
(function initActiveLink() {
  const current = window.location.pathname.split('/').pop() || 'index.html';
  qsa('.nav__link, .nav__mobile a').forEach(a => {
    const href = a.getAttribute('href') || '';
    if (href === current || (current === 'index.html' && href === '') || href === current.replace('.html', '')) {
      a.classList.add('active');
    }
  });
})();

/* ── Scroll reveal ───────────────────────────────────── */
(function initScrollReveal() {
  const elements = qsa('.reveal');
  if (!elements.length) return;

  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -60px 0px' });

  elements.forEach(el => io.observe(el));
})();

/* ── Contact form validation ─────────────────────────── */
(function initContactForm() {
  const form = qs('#contact-form');
  if (!form) return;

  /* Sanitize a string to prevent XSS if echoed anywhere */
  function sanitize(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function showError(fieldId, msg) {
    const field = qs(`#${fieldId}`);
    const err   = qs(`#${fieldId}-error`);
    if (!field || !err) return;
    field.classList.add('error');
    err.textContent = msg;
    err.classList.add('visible');
  }

  function clearError(fieldId) {
    const field = qs(`#${fieldId}`);
    const err   = qs(`#${fieldId}-error`);
    if (!field || !err) return;
    field.classList.remove('error');
    err.classList.remove('visible');
  }

  /* Live clear on input */
  ['cf-name', 'cf-email', 'cf-message'].forEach(id => {
    const el = qs(`#${id}`);
    if (el) el.addEventListener('input', () => clearError(id));
  });

  const EMAIL_RE = /^[^\s@]{1,64}@[^\s@]{1,255}\.[^\s@]{2,}$/;

  form.addEventListener('submit', e => {
    e.preventDefault();
    let valid = true;

    const nameEl    = qs('#cf-name');
    const emailEl   = qs('#cf-email');
    const messageEl = qs('#cf-message');

    const name    = nameEl?.value.trim() ?? '';
    const email   = emailEl?.value.trim() ?? '';
    const message = messageEl?.value.trim() ?? '';

    // Name
    if (!name) {
      showError('cf-name', 'Please enter your name.');
      valid = false;
    } else if (name.length > 120) {
      showError('cf-name', 'Name is too long (max 120 characters).');
      valid = false;
    }

    // Email
    if (!email) {
      showError('cf-email', 'Please enter your email address.');
      valid = false;
    } else if (!EMAIL_RE.test(email) || email.length > 254) {
      showError('cf-email', 'Please enter a valid email address.');
      valid = false;
    }

    // Message
    if (!message) {
      showError('cf-message', 'Please enter your message.');
      valid = false;
    } else if (message.length < 10) {
      showError('cf-message', 'Message is too short (min 10 characters).');
      valid = false;
    } else if (message.length > 2000) {
      showError('cf-message', 'Message is too long (max 2000 characters).');
      valid = false;
    }

    if (!valid) return;

    /* ── Send to Supabase Database (or fallback if unconfigured) ── */
    const btn     = qs('#cf-submit');
    const success = qs('#form-success');
    btn.disabled = true;
    btn.textContent = 'Sending…';

    const client = window.supabaseClient;

    if (client && window.SUPABASE_CONFIGURED) {
      client
        .from('contact_queries')
        .insert([{ name, email, message }])
        .then(({ data, error }) => {
          btn.disabled = false;
          btn.textContent = 'Send Message';
          if (error) {
            console.error('Supabase submission error:', error);
            showError('cf-message', 'Submission error: ' + (error.message || 'Check database RLS policy or table existence.'));
          } else {
            form.reset();
            if (success) {
              success.textContent = '✓ Message saved! We will be in touch within one business day.';
              success.classList.add('visible');
            }
          }
        })
        .catch(err => {
          console.error('Unexpected error:', err);
          btn.disabled = false;
          btn.textContent = 'Send Message';
          showError('cf-message', 'Connection failed (' + (err.message || 'Failed to fetch') + '). Check API Key or AdBlockers.');
        });
    } else {
      // Fallback mode if Supabase credentials are not yet populated in js/supabase-config.js
      setTimeout(() => {
        form.reset();
        btn.disabled = false;
        btn.textContent = 'Send Message';
        if (success) {
          success.textContent = '✓ Message received! (Demo mode: Please configure js/supabase-config.js to persist queries)';
          success.classList.add('visible');
        }
      }, 800);
    }
  });
})();
