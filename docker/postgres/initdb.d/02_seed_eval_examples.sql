-- Seed a few golden evaluation examples for offline testing.
-- Safe to run multiple times; inserts skip if the query already exists.

INSERT INTO eval_examples (query, gold_answer, expected_keywords, difficulty, category)
SELECT
    'What is the RBA''s inflation target?',
    'The RBA targets consumer price inflation of 2–3 percent on average over time.',
    '["2-3", "percent", "inflation", "target"]'::jsonb,
    'easy',
    'inflation'
WHERE NOT EXISTS (
    SELECT 1 FROM eval_examples WHERE query = 'What is the RBA''s inflation target?'
);

INSERT INTO eval_examples (query, gold_answer, expected_keywords, difficulty, category)
SELECT
    'Who sets Australia''s cash rate and how often is it reviewed?',
    'The Reserve Bank Board sets the cash rate target at its monetary policy meetings, typically held 11 times per year.',
    '["Reserve Bank Board", "cash rate", "meeting", "11"]'::jsonb,
    'medium',
    'policy'
WHERE NOT EXISTS (
    SELECT 1 FROM eval_examples WHERE query = 'Who sets Australia''s cash rate and how often is it reviewed?'
);

INSERT INTO eval_examples (query, gold_answer, expected_keywords, difficulty, category)
SELECT
    'What is the width of the cash rate corridor around the RBA''s target?',
    'The corridor is typically 50 basis points wide, with the deposit and lending rates set 25 basis points below and above the cash rate target.',
    '["50", "basis points", "corridor", "deposit", "lending"]'::jsonb,
    'medium',
    'operations'
WHERE NOT EXISTS (
    SELECT 1 FROM eval_examples WHERE query = 'What is the width of the cash rate corridor around the RBA''s target?'
);
