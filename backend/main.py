"""
Family Tree FastAPI Backend
Exposes REST API endpoints for the frontend and AI agent.
"""

import logging
import os
import random
import sys
from typing import Optional

from dotenv import load_dotenv
from falkordb import FalkorDB
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# --- Config ---
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", 6379))
GRAPH_NAME    = os.getenv("FALKORDB_GRAPH_NAME", "family_tree")

# --- App ---
app = FastAPI(
    title="Family Tree API",
    description="API for exploring the Veerapuram family tree",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB Connection ---
def get_graph():
    db    = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    graph = db.select_graph(GRAPH_NAME)
    return graph


# --- Helpers ---
def node_to_dict(node) -> dict:
    """Convert a FalkorDB node to a plain dictionary."""
    return dict(node.properties)


# --- Agent ---
from agent.agent import ask as ask_agent

# --- Models ---
class AskRequest(BaseModel):
    question: str


# --- Routes ---

@app.get("/")
def root():
    return {"message": "Family Tree API is running!"}


@app.get("/health")
def health():
    try:
        graph = get_graph()
        result = graph.query("MATCH (p:Person) RETURN count(p) AS total")
        total = result.result_set[0][0]
        return {"status": "healthy", "total_members": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Return interesting statistics about the family tree."""
    try:
        graph = get_graph()

        # Total members
        total = graph.query(
            "MATCH (p:Person) RETURN count(p) AS total"
        ).result_set[0][0]

        # Total relationships
        rels = graph.query(
            "MATCH ()-[r]->() RETURN count(r) AS total"
        ).result_set[0][0]

        # Males vs Females
        males = graph.query(
            "MATCH (p:Person {gender: 'Male'}) RETURN count(p)"
        ).result_set[0][0]

        females = graph.query(
            "MATCH (p:Person {gender: 'Female'}) RETURN count(p)"
        ).result_set[0][0]

        # Oldest person
        oldest = graph.query(
            """MATCH (p:Person)
               WHERE p.born <> '' AND p.died = ''
               RETURN p.full_name, p.born
               ORDER BY p.born ASC LIMIT 1"""
        ).result_set

        # Youngest person
        youngest = graph.query(
            """MATCH (p:Person)
               WHERE p.born <> '' AND p.died = ''
               RETURN p.full_name, p.born
               ORDER BY p.born DESC LIMIT 1"""
        ).result_set

        # Members who have passed
        deceased = graph.query(
            "MATCH (p:Person) WHERE p.died <> '' RETURN count(p)"
        ).result_set[0][0]

        return {
            "total_members":    total,
            "total_relationships": rels,
            "males":            males,
            "females":          females,
            "deceased":         deceased,
            "oldest":           {"name": oldest[0][0], "born": oldest[0][1]} if oldest else None,
            "youngest":         {"name": youngest[0][0], "born": youngest[0][1]} if youngest else None,
        }

    except Exception as e:
        log.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/members")
def get_all_members():
    """Return all family members."""
    try:
        graph   = get_graph()
        result  = graph.query("MATCH (p:Person) RETURN p ORDER BY p.full_name")
        members = [node_to_dict(row[0]) for row in result.result_set]
        return {"members": members, "total": len(members)}
    except Exception as e:
        log.error(f"Error getting members: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/members/search")
def search_members(name: str):
    """Search for family members by name (partial match)."""
    try:
        graph  = get_graph()
        result = graph.query(
            """MATCH (p:Person)
               WHERE toLower(p.full_name) CONTAINS toLower($name)
               RETURN p""",
            {"name": name}
        )
        members = [node_to_dict(row[0]) for row in result.result_set]
        return {"members": members, "total": len(members)}
    except Exception as e:
        log.error(f"Error searching members: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/members/random")
def get_random_member():
    """Return a random family member."""
    try:
        graph   = get_graph()
        result  = graph.query("MATCH (p:Person) RETURN p")
        all     = [node_to_dict(row[0]) for row in result.result_set]
        if not all:
            raise HTTPException(status_code=404, detail="No members found")
        return random.choice(all)
    except Exception as e:
        log.error(f"Error getting random member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/members/{full_name}")
def get_member(full_name: str):
    """Get a specific family member by full name."""
    try:
        graph  = get_graph()
        result = graph.query(
            "MATCH (p:Person {full_name: $name}) RETURN p",
            {"name": full_name}
        )
        if not result.result_set:
            raise HTTPException(status_code=404, detail=f"Member '{full_name}' not found")
        return node_to_dict(result.result_set[0][0])
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/members/{full_name}/connections")
def get_connections(full_name: str):
    """Get all immediate (first-order) connections for a person."""
    try:
        graph = get_graph()

        # Check person exists
        exists = graph.query(
            "MATCH (p:Person {full_name: $name}) RETURN p",
            {"name": full_name}
        )
        if not exists.result_set:
            raise HTTPException(status_code=404, detail=f"Member '{full_name}' not found")

        # Parents
        parents = graph.query(
            """MATCH (parent:Person)-[:PARENT_OF]->
               (p:Person {full_name: $name})
               RETURN parent""",
            {"name": full_name}
        )

        # Children
        children = graph.query(
            """MATCH (p:Person {full_name: $name})
               -[:PARENT_OF]->(child:Person)
               RETURN child""",
            {"name": full_name}
        )

        # Spouse
        spouse = graph.query(
            """MATCH (p:Person {full_name: $name})
               -[:SPOUSE_OF]-(s:Person)
               RETURN s""",
            {"name": full_name}
        )

        # Siblings (share at least one parent)
        siblings = graph.query(
            """MATCH (parent:Person)-[:PARENT_OF]->
               (p:Person {full_name: $name}),
               (parent)-[:PARENT_OF]->(sibling:Person)
               WHERE sibling.full_name <> $name
               RETURN DISTINCT sibling""",
            {"name": full_name}
        )

        return {
            "person":   full_name,
            "parents":  [node_to_dict(r[0]) for r in parents.result_set],
            "children": [node_to_dict(r[0]) for r in children.result_set],
            "spouse":   [node_to_dict(r[0]) for r in spouse.result_set],
            "siblings": [node_to_dict(r[0]) for r in siblings.result_set],
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting connections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(request: AskRequest):
    """Send a natural language question to the AI agent."""
    try:
        log.info(f"Question received: {request.question}")
        answer = ask_agent(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        log.error(f"Error asking question: {e}")
        raise HTTPException(status_code=500, detail=str(e))