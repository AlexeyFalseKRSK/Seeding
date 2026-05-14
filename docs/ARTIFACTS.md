# Local artifacts

This project intentionally keeps trained weights, datasets, generated reports,
and experiment outputs outside normal Git commits.

## What stays outside Git

- `models/` - YOLO/PyTorch weights (`*.pt`)
- `dataset/` - YOLO datasets and training splits
- `results/` - training/evaluation outputs
- `runs/` - Ultralytics run outputs
- `Photo/` - local demo/input materials
- `seeding/data/*.sqlite3` - local SQLite databases

These paths are already covered by `.gitignore`.

## Required runtime weights

The desktop app needs these two files by default:

| Purpose | Default path | Override |
| --- | --- | --- |
| Seedling detection | `models/bestDetectNew.pt` | `YOLO_WEIGHTS_PATH` or `seeding --weights` |
| Plant-part segmentation | `models/bestKlassSegFlip180.pt` | `YOLO_CLASSIFY_WEIGHTS_PATH` or `seeding --classify-weights` |

Keep the files in `models/` for local development, or point the environment
variables/CLI arguments to another storage location.

## Recommended storage

- Put release-ready model files in GitHub Releases, DVC, or another artifact store.
- Put datasets in DVC with remote storage such as S3, Google Drive, or an internal share.
- Record the exact artifact version used for a release in the release notes.
- Do not commit placeholder or zero-byte `.pt` files; they break startup checks and
  make it unclear which model is valid.

## MCP sanity check

`.mcp.json` currently declares:

- `memory` via `npx @modelcontextprotocol/server-memory`
- `git` via `uvx mcp-server-git`
- `sqlite` via `uvx mcp-server-sqlite`
- `filesystem` via `npx @modelcontextprotocol/server-filesystem`
- `obsidian` via `npx mcp-obsidian`

On this machine the JSON is valid, `npx` is available, `uvx.exe` exists, and the
configured SQLite path exists. If the project is moved to another machine, update
the absolute Windows paths in `.mcp.json`.
