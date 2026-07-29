/* =====================================================
   supabase-config.js — Fazal-e-Rashid Vision
   Supabase client configuration and initialization
   ===================================================== */

// REPLACE THESE WITH YOUR SUPABASE PROJECT CREDENTIALS
// Find these in your Supabase Dashboard -> Project Settings -> API

const SUPABASE_URL = 'https://mzetgzfkpsmitszaicnw.supabase.co'; 
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im16ZXRnemZrcHNtaXRzemFpY253Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODQ5Nzg1NTIsImV4cCI6MjEwMDU1NDU1Mn0.Y9Bn3YbD9Ft4CPdTTwhz5K311KB-QfJqI9lybBuylD8';
let supabaseClient = null;

if (typeof supabase !== 'undefined' && SUPABASE_URL && !SUPABASE_URL.includes('NEXT_PUBLIC')) {
  try {
    supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    console.log('✅ Supabase client initialized successfully.');
  } catch (err) {
    console.error('❌ Failed to initialize Supabase client:', err);
  }
} else {
  console.warn('⚠️ Supabase credentials not set or SDK not loaded yet. Form will operate in fallback mode.');
}

window.supabaseClient = supabaseClient;
window.SUPABASE_CONFIGURED = Boolean(supabaseClient);
