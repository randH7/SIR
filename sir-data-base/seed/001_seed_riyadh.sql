-- Seed data for Riyadh only
-- Safe re-run: wipe dynamic tables first (keeps lookup tables)
TRUNCATE TABLE
  prediction_signals,
  signals,
  predictions,
  incident_source_channels,
  incidents
RESTART IDENTITY;

-- =========================
-- INCIDENTS (last 24h)
-- =========================
INSERT INTO incidents (
  incident_id, city_id, title, domain_id, severity, status_id,
  occurred_at, updated_at, location
) VALUES
  (
    'inc-1001', 'riyadh', 'Traffic collision reported', 'traffic', 3, 'open',
    now() - interval '2 hours', now() - interval '1 hour 40 minutes',
    ST_GeogFromText('POINT(46.6753 24.7136)')
  ),
  (
    'inc-1002', 'riyadh', 'Suspected illegal gathering', 'crowd', 4, 'in_progress',
    now() - interval '6 hours', now() - interval '5 hours 30 minutes',
    ST_GeogFromText('POINT(46.7070 24.7743)')
  ),
  (
    'inc-1003', 'riyadh', 'Medical emergency request', 'health', 2, 'closed',
    now() - interval '20 hours', now() - interval '19 hours 50 minutes',
    ST_GeogFromText('POINT(46.7312 24.6892)')
  ),
  (
    'inc-1004', 'riyadh', 'Drug-related complaint', 'drugs', 5, 'open',
    now() - interval '12 hours', now() - interval '11 hours 45 minutes',
    ST_GeogFromText('POINT(46.6401 24.7424)')
  ),
  (
    'inc-1005', 'riyadh', 'Weapon report in public area', 'weapons', 5, 'in_progress',
    now() - interval '3 hours 30 minutes', now() - interval '3 hours 10 minutes',
    ST_GeogFromText('POINT(46.6215 24.6999)')
  );

-- incident -> source channels (junction)
INSERT INTO incident_source_channels (incident_id, source_channel_id) VALUES
  ('inc-1001', '911'),
  ('inc-1001', 'field_officers'),
  ('inc-1002', 'kolna_amn'),
  ('inc-1002', 'field_officers'),
  ('inc-1003', '911'),
  ('inc-1004', 'kolna_amn'),
  ('inc-1005', 'field_officers');

-- =========================
-- PREDICTIONS (future horizon)
-- =========================
INSERT INTO predictions (
  prediction_id, city_id, type_id, theme, confidence, risk_level_id,
  created_at, window_start, window_end, center, radius_meters, summary
) VALUES
  (
    'pred-2001', 'riyadh', 'current_hotspots', 'crash_risk', 0.76, 'high',
    now(),
    now() + interval '30 minutes',
    now() + interval '90 minutes',
    ST_GeogFromText('POINT(46.6849 24.7112)'),
    900,
    'Elevated crash risk in this zone for the next 60–90 minutes.'
  ),
  (
    'pred-2002', 'riyadh', 'overcrowding_risks', 'crowding', 0.68, 'medium',
    now(),
    now() + interval '60 minutes',
    now() + interval '180 minutes',
    ST_GeogFromText('POINT(46.7075 24.7740)'),
    1200,
    'Overcrowding risk expected near this area in the next 1–3 hours.'
  ),
  (
    'pred-2003', 'riyadh', 'congestion_waves', 'crash_risk', 0.54, 'medium',
    now(),
    now() + interval '15 minutes',
    now() + interval '75 minutes',
    ST_GeogFromText('POINT(46.6408 24.7420)'),
    1100,
    'Congestion wave may increase collision risk in the next hour.'
  );

-- =========================
-- SIGNALS (evidence)
-- =========================
INSERT INTO signals (
  signal_id, city_id, source_type_id, theme, confidence,
  window_start, window_end, center, radius_meters, payload
) VALUES
  (
    'sig-5001', 'riyadh', 'weather', 'crash_risk', 0.61,
    now() - interval '15 minutes',
    now() + interval '45 minutes',
    ST_GeogFromText('POINT(46.6830 24.7120)'),
    1500,
    jsonb_build_object('type', 'fog', 'visibility_m', 800)
  ),
  (
    'sig-5002', 'riyadh', 'history', 'crash_risk', 0.72,
    now() - interval '2 hours',
    now() + interval '6 hours',
    ST_GeogFromText('POINT(46.6860 24.7105)'),
    1200,
    jsonb_build_object('note', 'Historically high crash density zone')
  ),
  (
    'sig-5003', 'riyadh', 'social', 'crowding', 0.58,
    now() - interval '30 minutes',
    now() + interval '120 minutes',
    ST_GeogFromText('POINT(46.7074 24.7742)'),
    1800,
    jsonb_build_object('platform', 'x', 'keywords', jsonb_build_array('gathering', 'crowd'), 'language', 'ar')
  ),
  (
    'sig-5004', 'riyadh', 'events', 'crowding', 0.66,
    now(),
    now() + interval '240 minutes',
    ST_GeogFromText('POINT(46.7078 24.7736)'),
    2000,
    jsonb_build_object('event_type', 'festival', 'expected_attendance', 5000)
  ),
  (
    'sig-5005', 'riyadh', 'news', 'crash_risk', 0.44,
    now() - interval '60 minutes',
    now() + interval '180 minutes',
    ST_GeogFromText('POINT(46.6410 24.7418)'),
    1700,
    jsonb_build_object('headline', 'Road works causing slow traffic', 'source', 'local')
  );

-- prediction <-> signals (explanations)
INSERT INTO prediction_signals (prediction_id, signal_id, weight, summary) VALUES
  ('pred-2001', 'sig-5002', 0.35, 'High crash density zone'),
  ('pred-2001', 'sig-5001', 0.20, 'Reduced visibility due to fog'),
  ('pred-2001', 'sig-5005', 0.12, 'Road works increasing congestion'),

  ('pred-2002', 'sig-5004', 0.30, 'Large event expected attendance'),
  ('pred-2002', 'sig-5003', 0.18, 'Social chatter indicates crowd movement'),

  ('pred-2003', 'sig-5005', 0.25, 'Road works causing slow traffic'),
  ('pred-2003', 'sig-5002', 0.10, 'Historical crash-prone corridor');
