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
-- INCIDENTS (last 24h)  ✅ PAST ONLY
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
-- PREDICTIONS (2h → 10h) ✅ FUTURE WINDOWS ONLY
-- created_at is in the past (model ran recently)
-- =========================
INSERT INTO predictions (
  prediction_id, city_id, type_id, theme, confidence, risk_level_id,
  created_at, window_start, window_end, center, radius_meters, summary
) VALUES
  -- 1) Current Hotspot
  (
    'pred-2001', 'riyadh', 'current_hotspots', 'crash_risk', 0.78, 'high',
    now() - interval '6 minutes',
    now() + interval '2 hours',
    now() + interval '4 hours',
    ST_GeogFromText('POINT(46.6849 24.7112)'),
    900,
    'Current hotspot forecast: elevated crash risk expected within 2–4 hours.'
  ),

  -- 2) Congestion Waves
  (
    'pred-2002', 'riyadh', 'congestion_waves', 'crash_risk', 0.63, 'medium',
    now() - interval '7 minutes',
    now() + interval '2 hours 30 minutes',
    now() + interval '5 hours',
    ST_GeogFromText('POINT(46.6408 24.7420)'),
    1100,
    'Congestion wave likely to form and increase collision probability in 2.5–5 hours.'
  ),

  -- 3) Crime Clusters
  (
    'pred-2003', 'riyadh', 'crime_clusters', 'public_safety', 0.59, 'medium',
    now() - interval '8 minutes',
    now() + interval '3 hours',
    now() + interval '6 hours',
    ST_GeogFromText('POINT(46.6215 24.6999)'),
    1000,
    'Potential crime cluster formation based on recent patterns and nearby risk signals (3–6 hours).'
  ),

  -- 4) Overcrowding Risks
  (
    'pred-2004', 'riyadh', 'overcrowding_risks', 'crowding', 0.71, 'high',
    now() - interval '9 minutes',
    now() + interval '3 hours 30 minutes',
    now() + interval '7 hours',
    ST_GeogFromText('POINT(46.7075 24.7740)'),
    1400,
    'Overcrowding risk expected to rise near this area within 3.5–7 hours.'
  ),

  -- 5) Current Hotspot (secondary)
  (
    'pred-2005', 'riyadh', 'current_hotspots', 'weapons_risk', 0.52, 'medium',
    now() - interval '10 minutes',
    now() + interval '4 hours',
    now() + interval '6 hours 30 minutes',
    ST_GeogFromText('POINT(46.6550 24.7355)'),
    850,
    'Localized hotspot risk flagged for weapons-related activity window (4–6.5 hours).'
  ),

  -- 6) Congestion Waves (later horizon)
  (
    'pred-2006', 'riyadh', 'congestion_waves', 'traffic_disruption', 0.56, 'medium',
    now() - interval '11 minutes',
    now() + interval '6 hours',
    now() + interval '9 hours',
    ST_GeogFromText('POINT(46.7312 24.6892)'),
    1600,
    'Traffic disruption wave may expand across nearby corridors within 6–9 hours.'
  ),

  -- 7) Crime Clusters (late horizon)
  (
    'pred-2007', 'riyadh', 'crime_clusters', 'drugs_risk', 0.49, 'low',
    now() - interval '12 minutes',
    now() + interval '8 hours',
    now() + interval '10 hours',
    ST_GeogFromText('POINT(46.6401 24.7424)'),
    1200,
    'Low-confidence cluster signal for drugs-related risk emerging in 8–10 hours.'
  );

-- =========================
-- SIGNALS (evidence) ✅ FUTURE ONLY (aligned with 2h → 10h predictions)
-- =========================
INSERT INTO signals (
  signal_id, city_id, source_type_id, theme, confidence,
  window_start, window_end, center, radius_meters, payload
) VALUES
  -- Crash / traffic evidence
  (
    'sig-5001', 'riyadh', 'weather', 'crash_risk', 0.61,
    now() + interval '2 hours',
    now() + interval '4 hours',
    ST_GeogFromText('POINT(46.6830 24.7120)'),
    1500,
    jsonb_build_object('type', 'fog', 'visibility_m', 800)
  ),
  (
    'sig-5002', 'riyadh', 'history', 'crash_risk', 0.72,
    now() + interval '2 hours',
    now() + interval '10 hours',
    ST_GeogFromText('POINT(46.6860 24.7105)'),
    1200,
    jsonb_build_object('note', 'Historically high crash density zone')
  ),
  (
    'sig-5003', 'riyadh', 'news', 'traffic_disruption', 0.57,
    now() + interval '6 hours',
    now() + interval '9 hours',
    ST_GeogFromText('POINT(46.7310 24.6900)'),
    1700,
    jsonb_build_object('headline', 'Road works scheduled; lane reductions expected', 'source', 'local')
  ),
  (
    'sig-5004', 'riyadh', 'events', 'traffic_disruption', 0.52,
    now() + interval '6 hours 30 minutes',
    now() + interval '9 hours 30 minutes',
    ST_GeogFromText('POINT(46.7285 24.6888)'),
    1800,
    jsonb_build_object('event_type', 'sports', 'expected_attendance', 8000)
  ),

  -- Crowding evidence
  (
    'sig-5005', 'riyadh', 'social', 'crowding', 0.58,
    now() + interval '3 hours',
    now() + interval '6 hours',
    ST_GeogFromText('POINT(46.7074 24.7742)'),
    1800,
    jsonb_build_object('platform', 'x', 'keywords', jsonb_build_array('gathering', 'crowd'), 'language', 'ar')
  ),
  (
    'sig-5006', 'riyadh', 'events', 'crowding', 0.66,
    now() + interval '3 hours',
    now() + interval '8 hours',
    ST_GeogFromText('POINT(46.7078 24.7736)'),
    2000,
    jsonb_build_object('event_type', 'festival', 'expected_attendance', 5000)
  ),

  -- Public safety / weapons evidence
  (
    'sig-5007', 'riyadh', 'history', 'public_safety', 0.55,
    now() + interval '3 hours',
    now() + interval '6 hours',
    ST_GeogFromText('POINT(46.6212 24.7002)'),
    1400,
    jsonb_build_object('note', 'Pattern: repeated public disturbances in this zone')
  ),
  (
    'sig-5008', 'riyadh', 'social', 'weapons_risk', 0.47,
    now() + interval '4 hours',
    now() + interval '6 hours 30 minutes',
    ST_GeogFromText('POINT(46.6552 24.7352)'),
    1200,
    jsonb_build_object('platform', 'x', 'keywords', jsonb_build_array('weapon', 'shots'), 'language', 'ar')
  ),
  (
    'sig-5009', 'riyadh', 'news', 'weapons_risk', 0.44,
    now() + interval '4 hours 15 minutes',
    now() + interval '6 hours 45 minutes',
    ST_GeogFromText('POINT(46.6547 24.7360)'),
    1500,
    jsonb_build_object('headline', 'Increased patrols announced in nearby district', 'source', 'local')
  ),

  -- Drugs risk evidence
  (
    'sig-5010', 'riyadh', 'history', 'drugs_risk', 0.46,
    now() + interval '8 hours',
    now() + interval '10 hours',
    ST_GeogFromText('POINT(46.6400 24.7426)'),
    1600,
    jsonb_build_object('note', 'Trend: recurring late-night reports in this corridor')
  );

-- =========================
-- prediction <-> signals (explanations)
-- ✅ ALL FUTURE-ALIGNED
-- =========================
INSERT INTO prediction_signals (prediction_id, signal_id, weight, summary) VALUES
  -- pred-2001 (crash_risk, 2h–4h)
  ('pred-2001', 'sig-5002', 0.32, 'Historically high crash density zone'),
  ('pred-2001', 'sig-5001', 0.22, 'Fog reduces visibility during the risk window'),

  -- pred-2002 (crash_risk, 2.5h–5h)
  ('pred-2002', 'sig-5002', 0.18, 'Known crash-prone corridor overlaps congestion window'),
  ('pred-2002', 'sig-5001', 0.10, 'Weather conditions contribute to crash likelihood'),

  -- pred-2003 (public_safety, 3h–6h)
  ('pred-2003', 'sig-5007', 0.30, 'Recurring disturbances pattern in the same geo-area'),
  ('pred-2003', 'sig-5005', 0.12, 'Social chatter indicates movement/activity'),

  -- pred-2004 (crowding, 3.5h–7h)
  ('pred-2004', 'sig-5006', 0.34, 'Large event expected attendance'),
  ('pred-2004', 'sig-5005', 0.20, 'Social chatter indicates crowd movement'),

  -- pred-2005 (weapons_risk, 4h–6.5h)
  ('pred-2005', 'sig-5008', 0.18, 'Social indicators mention weapons-related keywords'),
  ('pred-2005', 'sig-5009', 0.12, 'News signal suggests increased patrol focus'),

  -- pred-2006 (traffic_disruption, 6h–9h)
  ('pred-2006', 'sig-5003', 0.26, 'Scheduled road works likely to slow traffic'),
  ('pred-2006', 'sig-5004', 0.14, 'Event traffic surge overlaps disruption window'),

  -- pred-2007 (drugs_risk, 8h–10h)
  ('pred-2007', 'sig-5010', 0.20, 'Historical trend suggests elevated risk in this time window');
