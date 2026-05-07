PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    login TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_logs_user_id_created_at
    ON user_logs (user_id, created_at DESC);

-- ── СЕССИИ АНАЛИЗА ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_session (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    source_path     TEXT NOT NULL,
    page_count      INTEGER,
    calibration_ppm REAL,
    report_path     TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session_user_id
    ON analysis_session (user_id, created_at DESC);

-- ── ОБНАРУЖЕННЫЕ ПРОРОСТКИ ────────────────────────────────────
CREATE TABLE IF NOT EXISTS detection (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            INTEGER NOT NULL REFERENCES analysis_session(id) ON DELETE CASCADE,
    page_index            INTEGER NOT NULL,
    object_index          INTEGER NOT NULL,
    bbox_x                REAL,
    bbox_y                REAL,
    bbox_w                REAL,
    bbox_h                REAL,
    confidence            REAL,
    rotation_deg          REAL NOT NULL DEFAULT 0,
    orientation_uncertain INTEGER NOT NULL DEFAULT 0,
    is_manual             INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_detection_session_id
    ON detection (session_id, page_index, object_index);

-- ── ЧАСТИ РАСТЕНИЯ ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS plant_part (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL REFERENCES detection(id) ON DELETE CASCADE,
    class_name   TEXT NOT NULL,
    confidence   REAL,
    bbox_x       REAL,
    bbox_y       REAL,
    bbox_w       REAL,
    bbox_h       REAL,
    polygon_json TEXT,
    is_manual    INTEGER NOT NULL DEFAULT 0
);

-- ── ИСТОРИЯ РУЧНЫХ ПРАВОК ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS edit_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    target_type  TEXT NOT NULL,
    target_id    INTEGER NOT NULL,
    field        TEXT NOT NULL,
    value_before TEXT,
    value_after  TEXT,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_edit_history_target
    ON edit_history (target_type, target_id, created_at DESC);
