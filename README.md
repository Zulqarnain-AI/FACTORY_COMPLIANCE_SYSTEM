# Factory Compliance & Alert Escalation System

## Architecture Overview

```
factory-compliance-system/
├── backend/
│   ├── src/
│   │   ├── policy/          # Module: Policy Parsing (Groq LLM)
│   │   ├── detection/       # Module 1: Vision Detection Engine
│   │   ├── severity/        # Module 2: Severity Categorization Matrix
│   │   ├── escalation/      # Module 3: Escalation Pipeline (WebSockets)
│   │   ├── reports/         # Module 4: Automated Report Generation
│   │   └── api/             # FastAPI routes
│   ├── main.py
│   ├── config.py
│   └── requirements.txt
├── frontend/                # Module 5: React Dashboard
│   ├── src/
│   │   ├── components/
│   │   └── App.jsx
│   └── package.json
|
└── compliance_policy.pdf
```

## Tech Stack
- **Vision**: YOLOv8 (Ultralytics) + OpenCV
- **LLM**: Groq API (llama-3.3-70b-versatile)
- **Backend**: FastAPI + WebSockets
- **Database**: MongoDB (Motor async driver)
- **Frontend**: React + Vite + TailwindCSS
- **Queue**: asyncio pub/sub for real-time escalation

## Setup Instructions

### 1. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Fill in GROQ_API_KEY and MONGODB_URI
uvicorn main:app --reload --port 8000
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev               # Runs on http://localhost:5173
```

### 3. YOLO Model
The system uses `yolov8n.pt` (auto-downloaded on first run) as a base.
For best results, fine-tune on factory-specific data using the Kaggle dataset.
See `backend/src/detection/README_TRAINING.md` for guidance.

## Severity Mapping (from Policy)
| Behavior Class | Policy Alert Level | Severity |
|---|---|---|
| Safe Walkway Violation | WARNING | HIGH |
| Unauthorized Intervention | CRITICAL SAFETY NOTICE | CRITICAL |
| Opened Panel Cover | WARNING | HIGH |
| Carrying Overload with Forklift | CRITICAL SAFETY NOTICE | CRITICAL |

## Escalation Routing
- LOW / MED → DB log only
- HIGH / CRIT → DB log + WebSocket real-time alert to dashboard
