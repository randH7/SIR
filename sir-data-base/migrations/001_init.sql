CREATE EXTENSION IF NOT EXISTS postgis;

-- =========================
-- LOOKUP TABLES
-- =========================

-- 1) Cities
CREATE TABLE IF NOT EXISTS cities (
  id TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT INTO cities (id, value) VALUES
  ('riyadh', 'Riyadh'),
  ('makkah', 'Makkah'),
  ('dammam', 'Dammam')
ON CONFLICT (id) DO NOTHING;

-- 2) Incident domains
CREATE TABLE IF NOT EXISTS incident_domains (
  id TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT INTO incident_domains (id, value) VALUES
  ('traffic', 'Traffic'),
  ('drugs', 'Drugs'),
  ('health', 'Health'),
  ('crowd', 'Crowd'),
  ('weapons', 'Weapons'),
  ('fire', 'Fire'),
  ('theft', 'Theft'),
  ('other', 'Other')
ON CONFLICT (id) DO NOTHING;

-- 3) Incident statuses
CREATE TABLE IF NOT EXISTS incident_statuses (
  id TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT INTO incident_statuses (id, value) VALUES
  ('open', 'Open'),
  ('closed', 'Closed'),
  ('in_progress', 'In Progress')
ON CONFLICT (id) DO NOTHING;

-- 4) Source channels
CREATE TABLE IF NOT EXISTS source_channels (
  id TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT INTO source_channels (id, value) VALUES
  ('911', '911'),
  ('kolna_amn', 'Kolna Amn'),
  ('field_officers', 'Field Officers')
ON CONFLICT (id) DO NOTHING;

-- 5) Prediction types
CREATE TABLE IF NOT EXISTS prediction_types (
  id TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT INTO prediction_types (id, value) VALUES
  ('current_hotspots', 'Current Hotspots'),
  ('congestion_waves', 'Congestion Waves'),
  ('crime_clusters', 'Crime Clusters'),
  ('overcrowding_risks', 'Overcrowding Risks')
ON CONFLICT (id) DO NOTHING;

-- 6) Signal source types
CREATE TABLE IF NOT EXISTS signal_source_types (
  id TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT INTO signal_source_types (id, value) VALUES
  ('social', 'Social Media'),
  ('news', 'News'),
  ('events', 'Events'),
  ('weather', 'Weather'),
  ('history', 'Historical Patterns'),
  ('cctv', 'CCTV'),
  ('iot', 'IoT Sensors'),
  ('other', 'Other')
ON CONFLICT (id) DO NOTHING;

-- 7) Risk levels
CREATE TABLE IF NOT EXISTS risk_levels (
  id TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  rank INT NOT NULL UNIQUE
);

INSERT INTO risk_levels (id, value, rank) VALUES
  ('low', 'Low', 1),
  ('medium', 'Medium', 2),
  ('high', 'High', 3),
  ('extreme', 'Extreme', 4)
ON CONFLICT (id) DO NOTHING;

-- =========================
-- MAIN TABLES
-- =========================

-- INCIDENTS
CREATE TABLE IF NOT EXISTS incidents (
  incident_id TEXT PRIMARY KEY,
  city_id TEXT NOT NULL REFERENCES cities(id),
  title TEXT NOT NULL,
  domain_id TEXT NOT NULL REFERENCES incident_domains(id),
  severity INT NOT NULL CHECK (severity BETWEEN 1 AND 5),
  status_id TEXT NOT NULL REFERENCES incident_statuses(id),
  occurred_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  location GEOGRAPHY(POINT, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS incidents_city_time_idx ON incidents (city_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS incidents_status_idx ON incidents (status_id);
CREATE INDEX IF NOT EXISTS incidents_location_gix ON incidents USING GIST (location);

-- INCIDENT <-> SOURCE CHANNELS (junction)
CREATE TABLE IF NOT EXISTS incident_source_channels (
  incident_id TEXT NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
  source_channel_id TEXT NOT NULL REFERENCES source_channels(id),
  PRIMARY KEY (incident_id, source_channel_id)
);

CREATE INDEX IF NOT EXISTS incident_source_channels_source_idx
  ON incident_source_channels (source_channel_id);

-- PREDICTIONS
CREATE TABLE IF NOT EXISTS predictions (
  prediction_id TEXT PRIMARY KEY,
  city_id TEXT NOT NULL REFERENCES cities(id),
  type_id TEXT NOT NULL REFERENCES prediction_types(id),
  theme TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  risk_level_id TEXT NOT NULL REFERENCES risk_levels(id),
  created_at TIMESTAMPTZ NOT NULL,
  window_start TIMESTAMPTZ NOT NULL,
  window_end TIMESTAMPTZ NOT NULL,
  center GEOGRAPHY(POINT, 4326) NOT NULL,
  radius_meters DOUBLE PRECISION NOT NULL CHECK (radius_meters > 0),
  summary TEXT
);

CREATE INDEX IF NOT EXISTS predictions_city_created_idx ON predictions (city_id, created_at DESC);
CREATE INDEX IF NOT EXISTS predictions_window_idx ON predictions (window_start, window_end);
CREATE INDEX IF NOT EXISTS predictions_center_gix ON predictions USING GIST (center);

-- SIGNALS
CREATE TABLE IF NOT EXISTS signals (
  signal_id TEXT PRIMARY KEY,
  city_id TEXT NOT NULL REFERENCES cities(id),
  source_type_id TEXT NOT NULL REFERENCES signal_source_types(id),
  theme TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  window_start TIMESTAMPTZ NOT NULL,
  window_end TIMESTAMPTZ NOT NULL,
  center GEOGRAPHY(POINT, 4326) NOT NULL,
  radius_meters DOUBLE PRECISION NOT NULL CHECK (radius_meters > 0),
  payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS signals_city_window_idx ON signals (city_id, window_start, window_end);
CREATE INDEX IF NOT EXISTS signals_center_gix ON signals USING GIST (center);

-- PREDICTION <-> SIGNALS (explainability mapping)
CREATE TABLE IF NOT EXISTS prediction_signals (
  prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
  signal_id TEXT NOT NULL REFERENCES signals(signal_id) ON DELETE CASCADE,
  weight DOUBLE PRECISION NOT NULL CHECK (weight BETWEEN 0 AND 1),
  summary TEXT,
  PRIMARY KEY (prediction_id, signal_id)
);
