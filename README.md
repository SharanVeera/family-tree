# 🌳 Veerapuram Family Tree

An AI and Graph-Powered Family Tree Browser built with:
- **FalkorDB** — Knowledge graph database
- **FastAPI** — REST API backend
- **LangGraph + Groq** — AI agent for natural language queries
- **Vanilla HTML/JS** — Frontend UI

## Setup

### Prerequisites
- Python 3.10+
- Docker Desktop

### Installation
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/family-tree.git
cd family-tree

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy env file and fill in your keys
cp .env.example .env
```

### Running the Project
```bash
# Start FalkorDB
docker-compose up -d

# Load data
python pipeline/load_graph.py

# Start backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Open frontend/index.html in your browser
```

## Project Structure
```
family-tree/
├── data/               # CSV files
├── pipeline/           # ETL pipeline
├── backend/            # FastAPI server
├── agent/              # LangGraph AI agent
├── frontend/           # Web UI
├── tests/              # Tests
└── docker-compose.yml
```