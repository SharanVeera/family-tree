"""
Data Pipeline: CSV → FalkorDB Graph
Loads family members as nodes and relationships as edges.
"""

import csv
import logging
import os
import sys
from dotenv import load_dotenv
from falkordb import FalkorDB

load_dotenv()

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# --- Config ---
FALKORDB_HOST       = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT       = int(os.getenv("FALKORDB_PORT", 6379))
GRAPH_NAME          = os.getenv("FALKORDB_GRAPH_NAME", "family_tree")
MEMBERS_CSV         = os.path.join(os.path.dirname(__file__), "../data/family_members.csv")
RELATIONSHIPS_CSV   = os.path.join(os.path.dirname(__file__), "../data/family_relationships.csv")


def connect_to_db() -> FalkorDB:
    """Connect to FalkorDB instance."""
    log.info(f"Connecting to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}")
    db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    log.info("✅ Connected to FalkorDB")
    return db


def clear_graph(graph):
    """Delete all existing nodes and edges in the graph."""
    log.info(f"Clearing existing graph '{GRAPH_NAME}'...")
    graph.query("MATCH (n) DETACH DELETE n")
    log.info("✅ Graph cleared")


def load_members(graph):
    """Load family members CSV as Person nodes."""
    log.info(f"Loading family members from {MEMBERS_CSV}")
    count = 0

    with open(MEMBERS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from all fields
            row = {k: v.strip() for k, v in row.items()}

            # Build the cypher query with all person properties
            query = """
                CREATE (p:Person {
                    full_name:      $full_name,
                    first_name:     $first_name,
                    middle_name:    $middle_name,
                    maiden_name:    $maiden_name,
                    last_name:      $last_name,
                    born:           $born,
                    died:           $died,
                    age:            $age,
                    gender:         $gender,
                    ethnicity1:     $ethnicity1,
                    ethnicity2:     $ethnicity2,
                    notes:          $notes
                })
            """
            params = {
                "full_name":    row.get("Full Name", ""),
                "first_name":   row.get("First Name", ""),
                "middle_name":  row.get("Middle Name", ""),
                "maiden_name":  row.get("Maiden Name", ""),
                "last_name":    row.get("Last Name", ""),
                "born":         row.get("Born", ""),
                "died":         row.get("Died", ""),
                "age":          row.get("Age", ""),
                "gender":       row.get("Gender", ""),
                "ethnicity1":   row.get("Ethnicity 1", ""),
                "ethnicity2":   row.get("Ethnicity 2", ""),
                "notes":        row.get("Notes", ""),
            }

            graph.query(query, params)
            count += 1
            log.info(f"  Created Person node: {row['Full Name']}")

    log.info(f"✅ Loaded {count} Person nodes")
    return count


def load_relationships(graph):
    """Load family relationships CSV as edges between Person nodes."""
    log.info(f"Loading relationships from {RELATIONSHIPS_CSV}")
    count = 0
    skipped = 0

    with open(RELATIONSHIPS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            person_a  = row.get("Person A", "").strip()
            related   = row.get("Related to", "").strip().upper()
            person_b  = row.get("Person B", "").strip()

            if not person_a or not related or not person_b:
                log.warning(f"  Skipping incomplete row: {row}")
                skipped += 1
                continue

            # Map relationship type to edge label
            # Father → PARENT_OF (with gender context on the node itself)
            # Mother → PARENT_OF
            # Spouse → SPOUSE_OF
            if related == "FATHER":
                edge_label = "PARENT_OF"
            elif related == "MOTHER":
                edge_label = "PARENT_OF"
            elif related == "SPOUSE":
                edge_label = "SPOUSE_OF"
            elif related == "CHILD":
                edge_label = "PARENT_OF"
                # Flip direction: Person A is the child → make Person B parent
                person_a, person_b = person_b, person_a
            elif related == "SIBLING":
                edge_label = "SIBLING_OF"
            else:
                log.warning(f"  Unknown relationship type '{related}', skipping")
                skipped += 1
                continue

            query = f"""
                MATCH (a:Person {{full_name: $person_a}})
                MATCH (b:Person {{full_name: $person_b}})
                CREATE (a)-[:{edge_label} {{type: $rel_type}}]->(b)
            """
            params = {
                "person_a": person_a,
                "person_b": person_b,
                "rel_type": related,
            }

            result = graph.query(query, params)
            count += 1
            log.info(f"  Created edge: {person_a} -[{edge_label}]-> {person_b}")

    log.info(f"✅ Loaded {count} relationships ({skipped} skipped)")
    return count


def verify_graph(graph):
    """Run basic checks to confirm the graph loaded correctly."""
    log.info("Verifying graph contents...")

    result = graph.query("MATCH (p:Person) RETURN count(p) AS total")
    total_nodes = result.result_set[0][0]

    result = graph.query("MATCH ()-[r]->() RETURN count(r) AS total")
    total_edges = result.result_set[0][0]

    result = graph.query("MATCH ()-[r:PARENT_OF]->() RETURN count(r) AS total")
    parent_edges = result.result_set[0][0]

    result = graph.query("MATCH ()-[r:SPOUSE_OF]->() RETURN count(r) AS total")
    spouse_edges = result.result_set[0][0]

    log.info(f"  👥 Total Person nodes : {total_nodes}")
    log.info(f"  🔗 Total edges        : {total_edges}")
    log.info(f"  👨‍👧 PARENT_OF edges    : {parent_edges}")
    log.info(f"  💑 SPOUSE_OF edges    : {spouse_edges}")

    return {
        "nodes": total_nodes,
        "edges": total_edges,
        "parent_edges": parent_edges,
        "spouse_edges": spouse_edges,
    }


def run_pipeline():
    """Main pipeline entry point."""
    log.info("=" * 50)
    log.info("🚀 Starting Family Tree Data Pipeline")
    log.info("=" * 50)

    db      = connect_to_db()
    graph   = db.select_graph(GRAPH_NAME)

    clear_graph(graph)
    members_count = load_members(graph)
    rels_count    = load_relationships(graph)
    stats         = verify_graph(graph)

    log.info("=" * 50)
    log.info("✅ Pipeline complete!")
    log.info(f"   Members loaded      : {members_count}")
    log.info(f"   Relationships loaded: {rels_count}")
    log.info(f"   Graph nodes         : {stats['nodes']}")
    log.info(f"   Graph edges         : {stats['edges']}")
    log.info("=" * 50)


if __name__ == "__main__":
    run_pipeline()