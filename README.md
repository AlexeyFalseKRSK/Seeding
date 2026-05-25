# Seeding

Desktop application for seedling image analysis. Seeding helps an operator load
photos or PDF scans, find seedlings with YOLO, segment plant parts, review and
correct annotations, save analysis sessions, and generate PDF reports.

## What It Does

- Loads image files and PDF documents as project pages.
- Detects seedlings on the current page or across the whole project.
- Segments seedling parts such as root, stem, and inflorescence.
- Shows bounding boxes and pixel masks on the image canvas.
- Supports manual bbox creation, bbox editing, deletion, and re-segmentation.
- Rotates pages and crops while keeping bbox coordinates and crops in sync.
- Saves and restores sessions in a local SQLite database.
- Preserves page rotation, crop rotation, manual flags, polygon masks, and bitmap masks.
- Generates PDF reports with annotated images and measurement tables.
- Supports local users, login, audit logs, and per-user session lists.

## Tech Stack

- Python 3.10+
- PyQt5 for the desktop UI
- Ultralytics YOLO for detection and segmentation
- OpenCV, NumPy, Pillow for image processing
- PyMuPDF for PDF rendering
- ReportLab for PDF report generation
- SQLite for users, audit logs, sessions, detections, and plant parts
- pytest and ruff for verification

## Repository Layout

```text
Seeding/
├── seeding/
│   ├── data/               # SQLite schema and local database location
│   ├── resources/          # icons and QSS styles
│   ├── ui/                 # PyQt widgets and main window
│   ├── utils/              # geometry and path helpers
│   ├── auth.py             # password hashing and verification
│   ├── config.py           # paths, thresholds, UI constants
│   ├── database.py         # SQLite schema initialization and CRUD
│   ├── image_loader.py     # image/PDF loading and session image restore
│   ├── inference.py        # YOLO backend abstraction
│   ├── main.py             # application entry point
│   ├── manage_users.py     # user management CLI
│   ├── mask_refiner.py     # mask cleanup and bitmap/contour helpers
│   ├── models.py           # AppState, ObjectImage, AllClassImage
│   ├── report.py           # PDF report generation
│   ├── services.py         # detection, segmentation, rotation logic
│   ├── session_service.py  # save/load analysis sessions
│   └── user_service.py     # users and audit log service
├── tests/                  # pytest suite
├── models/                 # local YOLO weights, not tracked by git
├── Photo/                  # local working photos, not tracked by git
├── README.md
└── pyproject.toml
```

## Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest ruff
```

The project expects YOLO `.pt` files to exist locally. By default:

```text
models/bestDetectNew.pt
models/bestKlassSegFlip180.pt
```

These model files are intentionally not tracked by git.

## Configuration

Model paths can be passed through CLI arguments:

```powershell
seeding --weights models\bestDetectNew.pt --classify-weights models\bestKlassSegFlip180.pt
```

Or through environment variables:

```powershell
$env:YOLO_WEIGHTS_PATH = "E:\models\bestDetectNew.pt"
$env:YOLO_CLASSIFY_WEIGHTS_PATH = "E:\models\bestKlassSegFlip180.pt"
$env:SEEDING_DB_PATH = "E:\data\seeding.sqlite3"
seeding
```

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `YOLO_WEIGHTS_PATH` | Detector weights path |
| `YOLO_CLASSIFY_WEIGHTS_PATH` | Segmentation/classification weights path |
| `YOLO_DEVICE` | YOLO device, for example `auto`, `cpu`, `0` |
| `SEEDING_DB_PATH` | SQLite database path |

## User Management

Create at least one local user before opening the app:

```powershell
seeding-users create admin
seeding-users list
seeding-users set-password admin
seeding-users delete admin
```

Passwords are stored as PBKDF2-HMAC-SHA256 hashes. Plain passwords are not saved.

## Running The App

```powershell
seeding
```

Typical workflow:

1. Log in.
2. Open an image or PDF.
3. Rotate the page if needed.
4. Run seedling detection.
5. Run segmentation for new or selected seedlings.
6. Review boxes, masks, statistics, and details.
7. Correct annotations manually if needed.
8. Save the session.
9. Restore the session later from the session picker.
10. Generate a PDF report.

## Session Persistence

Sessions are stored in SQLite. A saved session contains:

- source file paths for every page;
- page rotation state;
- calibration value;
- detected seedling bboxes;
- crop rotation state;
- orientation uncertainty flag;
- manual annotation flags;
- plant part bboxes;
- polygon masks;
- bitmap masks for accurate thin structures;
- last generated report path.

When a session is restored, Seeding reloads the original source files, reapplies
saved page rotations, rebuilds crop images from saved bboxes, and restores saved
part masks. If a source file is missing, the session metadata is still loaded and
the UI shows which source path could not be found.

## Main Controls

| Action | Shortcut / Place |
| --- | --- |
| Open files | `Ctrl+O` |
| Add files | `Ctrl+Shift+O` |
| Open saved session | `Ctrl+Alt+O` |
| Detect on current page | `Ctrl+F` |
| Detect on all/new pages | toolbar split button |
| Segment selected/new/all seedlings | toolbar split button or Analysis menu |
| Rotate page or selected crop | `Ctrl+R` |
| Save session | `Ctrl+S` |
| Generate report | `Ctrl+P` |
| Zoom in/out | `Ctrl++`, `Ctrl+-` |
| Fit image | `Ctrl+0` |
| Measurement mode | `M` |
| Delete selected annotation | `Delete` in bbox edit mode |
| Cancel drawing/measurement | `Esc` |

## Testing And Quality Checks

Run the same checks before pushing changes:

```powershell
python -m compileall -q seeding tests
python -m ruff check seeding tests
python -m pytest tests -q
```

Current suite covers:

- authentication and local users;
- SQLite schema and migrations;
- session save/load/restore;
- page rotation persistence;
- bitmap mask persistence;
- manual annotation flags;
- image/PDF loading helpers;
- inference adapters;
- mask refinement and contour conversion;
- report generation;
- UI behavior around the main window, tree, statistics, dialogs, and annotations.

## Development Notes

- Keep generated files, local photos, model weights, and temporary artifacts out of git.
- Prefer focused tests for every session, rotation, mask, and annotation change.
- Be careful with `seeding/ui/main_window.py`: it owns a lot of UI orchestration and is easy to break during merges.
- Session restore depends on source files still existing at their saved paths.
- If a database already exists, `database.initialize_database()` applies lightweight migrations for new columns.

## Git Hygiene

The intended tracked project surface is:

```text
seeding/
tests/
README.md
pyproject.toml
.gitignore
.gitattributes
```

Local folders such as `models/`, `Photo/`, caches, reports, and database files should stay untracked unless there is a deliberate reason to add them.
