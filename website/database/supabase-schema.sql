-- =============================================================
-- Fazal-e-Rashid Vision — Supabase Contact Queries Schema & RLS
-- =============================================================
-- HOW TO RUN THIS:
-- 1. Go to https://supabase.com/dashboard
-- 2. Open your project
-- 3. Click "SQL Editor" in the left sidebar
-- 4. Paste this entire file and click "Run"
-- =============================================================

-- Step 1: Create the table (safe to re-run)
CREATE TABLE IF NOT EXISTS public.contact_queries (
    id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now() NOT NULL,
    name        TEXT        NOT NULL,
    email       TEXT        NOT NULL,
    message     TEXT        NOT NULL,
    status      TEXT        DEFAULT 'new' NOT NULL
);

-- Step 2: Enable Row Level Security
ALTER TABLE public.contact_queries ENABLE ROW LEVEL SECURITY;

-- Step 3: DROP all existing policies first (prevents conflicts on re-run)
DROP POLICY IF EXISTS "Allow public form submissions"   ON public.contact_queries;
DROP POLICY IF EXISTS "Allow admins to read queries"    ON public.contact_queries;
DROP POLICY IF EXISTS "Allow admins to update queries"  ON public.contact_queries;
DROP POLICY IF EXISTS "Allow admins to delete queries"  ON public.contact_queries;

-- Step 4: Allow ANY visitor (anon role) to INSERT — this is what the contact form uses
CREATE POLICY "Allow public form submissions"
ON public.contact_queries
FOR INSERT
TO anon
WITH CHECK (true);

-- Step 5: Allow logged-in admins to SELECT (read) all rows
CREATE POLICY "Allow admins to read queries"
ON public.contact_queries
FOR SELECT
TO authenticated
USING (true);

-- Step 6: Allow logged-in admins to UPDATE (change status)
CREATE POLICY "Allow admins to update queries"
ON public.contact_queries
FOR UPDATE
TO authenticated
USING (true);

-- Step 7: Allow logged-in admins to DELETE rows
CREATE POLICY "Allow admins to delete queries"
ON public.contact_queries
FOR DELETE
TO authenticated
USING (true);

-- =============================================================
-- After running this SQL:
-- 1. Go to Authentication → Users → "Add User"
-- 2. Create your admin account, e.g.:
--    Email:    admin@fazlerasheed.com
--    Password: (something strong)
-- 3. Use those credentials to log in at /admin
-- =============================================================
