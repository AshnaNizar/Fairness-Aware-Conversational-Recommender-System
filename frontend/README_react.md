# FA-CRS React + FastAPI Setup

## One-time setup

```bash
# 1. Install Python deps (add to your existing env)
pip install fastapi uvicorn python-multipart

# 2. Install Node deps
cd frontend
npm install
cd ..
```

## Development (two terminals)

**Terminal 1 — Python backend:**
```bash
uvicorn api_server:app --reload --port 7860
```

**Terminal 2 — React frontend:**
```bash
cd frontend
npm run dev
# Opens at http://localhost:3000
```

## Production (serve everything from FastAPI)

```bash
cd frontend
npm run build        # outputs to frontend/dist/
cd ..
uvicorn api_server:app --port 7860
# Visit http://localhost:7860
```

## File structure

```
project/
├── api_server.py          ← FastAPI backend (all ML logic)
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx        ← Full React UI
│       ├── index.css
│       └── assets/
│           ├── logo.jpg   ← M logo
│           └── avatar.jpg ← User persona icon
├── data/                  ← ratings.csv, movies_enriched.csv
└── outputs/kg/            ← best_model_kg.pt, candidate_pools.pkl
```

## API endpoints

| Method | Path           | Description                        |
|--------|----------------|------------------------------------|
| GET    | /random_user   | Load a random user + recommendations |
| POST   | /chat          | Send a message, get reply + updated recs |

The Gradio file (`gradio_app_chat_latest.py`) is no longer needed but kept as backup.
