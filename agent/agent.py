"""
Family Tree AI Agent
Uses LangGraph + Groq to answer natural language questions
about the family tree by generating and executing Cypher queries.
All queries are logged to logs/query_log.jsonl for debugging.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from falkordb import FalkorDB
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# --- Logging Setup ---
LOG_DIR        = os.path.join(os.path.dirname(__file__), "../logs")
LOG_FILE       = os.path.join(LOG_DIR, "agent.log")
QUERY_LOG_FILE = os.path.join(LOG_DIR, "query_log.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)


def log_query(question: str, cypher: str, db_result: list,
              answer: str, is_relation_query: bool,
              path_data: dict = None, error: str = None):
    """
    Write one structured JSON entry per agent interaction to query_log.jsonl.
    Each line is a complete record — easy to grep, parse, or tail for debugging.
    """
    entry = {
        "timestamp":         datetime.now().isoformat(),
        "question":          question,
        "is_relation_query": is_relation_query,
        "cypher":            cypher,
        "path_data":         path_data or {},
        "db_result":         db_result,
        "answer":            answer,
        "error":             error,
    }
    try:
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info(f"📝 Query logged → {QUERY_LOG_FILE}")
    except Exception as e:
        log.error(f"Failed to write query log: {e}")


# --- Config ---
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", 6379))
GRAPH_NAME    = os.getenv("FALKORDB_GRAPH_NAME", "family_tree")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")


# --- DB ---
def get_graph():
    db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    return db.select_graph(GRAPH_NAME)


def run_cypher(query: str) -> list:
    try:
        graph  = get_graph()
        result = graph.query(query)
        rows   = []
        for row in result.result_set:
            parsed = []
            for item in row:
                if hasattr(item, 'properties'):
                    parsed.append(dict(item.properties))
                elif hasattr(item, 'nodes'):
                    parsed.append({
                        "nodes": [dict(n.properties) for n in item.nodes],
                        "edges": [r.type for r in item.relationships]
                    })
                else:
                    parsed.append(item)
            rows.append(parsed)
        return rows
    except Exception as e:
        log.error(f"Cypher error: {e}")
        return [{"error": str(e)}]


def get_path_between(person_a: str, person_b: str) -> dict:
    try:
        graph = get_graph()
        path_result = graph.query(f"""
            MATCH path = shortestPath(
                (a:Person {{full_name: "{person_a}"}})-[*]-(b:Person {{full_name: "{person_b}"}})
            )
            RETURN path
        """)
        if not path_result.result_set:
            return {"found": False}
        path  = path_result.result_set[0][0]
        nodes = [dict(n.properties) for n in path.nodes]
        edges = [r.type for r in path.relationships]
        steps = []
        for i, edge_type in enumerate(edges):
            from_node  = nodes[i]['full_name']
            to_node    = nodes[i+1]['full_name']
            dir_result = graph.query(f"""
                MATCH (a:Person {{full_name: "{from_node}"}})-[r]->(b:Person {{full_name: "{to_node}"}})
                RETURN type(r)
            """)
            steps.append({
                "from":        from_node,
                "from_gender": nodes[i].get('gender', ''),
                "edge":        dir_result.result_set[0][0] if dir_result.result_set else edge_type,
                "direction":   "forward" if dir_result.result_set else "reverse",
                "to":          to_node,
                "to_gender":   nodes[i+1].get('gender', '')
            })
        return {
            "found":       True,
            "person_a":    person_a,
            "person_b":    person_b,
            "path_length": len(edges),
            "nodes":       [n['full_name'] for n in nodes],
            "steps":       steps
        }
    except Exception as e:
        log.error(f"Path error: {e}")
        return {"found": False, "error": str(e)}


# --- Agent State ---
class AgentState(TypedDict):
    messages:          Annotated[list[AnyMessage], add_messages]
    question:          str
    cypher:            str
    db_result:         list
    path_data:         dict
    answer:            str
    is_relation_query: bool


SYSTEM_PROMPT = """You are a family tree assistant for the Veerapuram family.
You answer questions by writing Cypher queries for a FalkorDB graph database.

GRAPH SCHEMA:
Nodes: Person { full_name, first_name, last_name, gender, born, died, age, ethnicity1, ethnicity2, notes }
Edges:
  (parent:Person)-[:PARENT_OF]->(child:Person)
  (person:Person)-[:SPOUSE_OF]-(spouse:Person)

RELATIONSHIP PATTERNS:

PARENTS:
MATCH (parent:Person)-[:PARENT_OF]->(p:Person {full_name: $name}) RETURN parent.full_name, parent.gender

CHILDREN:
MATCH (p:Person {full_name: $name})-[:PARENT_OF]->(child:Person) RETURN child.full_name, child.gender

SPOUSE:
MATCH (p:Person {full_name: $name})-[:SPOUSE_OF]-(s:Person) RETURN s.full_name

SIBLINGS:
MATCH (parent:Person)-[:PARENT_OF]->(p:Person {full_name: $name}),(parent)-[:PARENT_OF]->(sibling:Person)
WHERE sibling.full_name <> $name RETURN DISTINCT sibling.full_name, sibling.gender

GRANDPARENTS:
MATCH (gp:Person)-[:PARENT_OF]->(:Person)-[:PARENT_OF]->(p:Person {full_name: $name})
RETURN DISTINCT gp.full_name, gp.gender

GRANDCHILDREN:
MATCH (p:Person {full_name: $name})-[:PARENT_OF]->(:Person)-[:PARENT_OF]->(gc:Person)
RETURN DISTINCT gc.full_name

GREAT GRANDPARENTS:
MATCH (ggp:Person)-[:PARENT_OF]->(:Person)-[:PARENT_OF]->(:Person)-[:PARENT_OF]->(p:Person {full_name: $name})
RETURN DISTINCT ggp.full_name

AUNTS & UNCLES:
MATCH (gp:Person)-[:PARENT_OF]->(parent:Person)-[:PARENT_OF]->(p:Person {full_name: $name}),(gp)-[:PARENT_OF]->(aunt_uncle:Person)
WHERE aunt_uncle.full_name <> parent.full_name RETURN DISTINCT aunt_uncle.full_name, aunt_uncle.gender

NIECES & NEPHEWS:
MATCH (parent:Person)-[:PARENT_OF]->(p:Person {full_name: $name}),(parent)-[:PARENT_OF]->(sibling:Person),(sibling)-[:PARENT_OF]->(niece_nephew:Person)
WHERE sibling.full_name <> $name RETURN DISTINCT niece_nephew.full_name, niece_nephew.gender

FIRST COUSINS:
MATCH (gp:Person)-[:PARENT_OF]->(parent:Person)-[:PARENT_OF]->(p:Person {full_name: $name}),(gp)-[:PARENT_OF]->(aunt_uncle:Person)-[:PARENT_OF]->(cousin:Person)
WHERE aunt_uncle.full_name <> parent.full_name RETURN DISTINCT cousin.full_name

SECOND COUSINS:
MATCH (ggp:Person)-[:PARENT_OF]->(gp1:Person)-[:PARENT_OF]->(parent:Person)-[:PARENT_OF]->(p:Person {full_name: $name}),(ggp)-[:PARENT_OF]->(gp2:Person)-[:PARENT_OF]->(aunt_uncle:Person)-[:PARENT_OF]->(first_cousin:Person)-[:PARENT_OF]->(second_cousin:Person)
WHERE gp2.full_name <> gp1.full_name RETURN DISTINCT second_cousin.full_name

PARENTS-IN-LAW:
MATCH (p:Person {full_name: $name})-[:SPOUSE_OF]-(spouse:Person),(parent_in_law:Person)-[:PARENT_OF]->(spouse)
RETURN DISTINCT parent_in_law.full_name, parent_in_law.gender

SIBLINGS-IN-LAW:
MATCH (p:Person {full_name: $name})-[:SPOUSE_OF]-(spouse:Person),(parent:Person)-[:PARENT_OF]->(spouse),(parent)-[:PARENT_OF]->(sibling_in_law:Person)
WHERE sibling_in_law.full_name <> spouse.full_name RETURN DISTINCT sibling_in_law.full_name

CHILDREN-IN-LAW:
MATCH (p:Person {full_name: $name})-[:PARENT_OF]->(child:Person),(child)-[:SPOUSE_OF]-(child_in_law:Person)
RETURN DISTINCT child_in_law.full_name, child_in_law.gender

ALL ANCESTORS:
MATCH (ancestor:Person)-[:PARENT_OF*]->(p:Person {full_name: $name}) RETURN DISTINCT ancestor.full_name

ALL DESCENDANTS:
MATCH (p:Person {full_name: $name})-[:PARENT_OF*]->(descendant:Person) RETURN DISTINCT descendant.full_name

STATISTICS:
Count over age X: MATCH (p:Person) WHERE toInteger(p.age) > 50 AND p.died = "" RETURN count(p)
Males vs females: MATCH (p:Person) RETURN p.gender, count(p)
Unmarried over 21: MATCH (p:Person) WHERE toInteger(p.age) > 21 AND p.died = "" AND NOT (p)-[:SPOUSE_OF]-() RETURN p.full_name, p.age, p.gender
Living outside India: MATCH (p:Person) WHERE p.notes CONTAINS "Lives in" RETURN p.full_name, p.notes

SPECIAL CASE:
If the question asks "how are X and Y related" or "what is the relationship between X and Y",
return this exact token and nothing else: RELATIONSHIP_QUERY

RULES:
1. Use full_name to match people
2. SPOUSE_OF is undirected — use -[:SPOUSE_OF]- no arrow
3. PARENT_OF is directed — (parent)-[:PARENT_OF]->(child)
4. age/born/died are strings — use toInteger(p.age) for math
5. p.died = "" means alive
6. Always return full_name
7. Use DISTINCT
8. Return ONLY raw Cypher or RELATIONSHIP_QUERY — no markdown, no backticks
"""

RELATIONSHIP_INTERPRETER_PROMPT = """You are an expert at interpreting family relationships.

Given a path through a family tree, state what Person B is to Person A.

PATH INTERPRETATION:
- forward PARENT_OF means: the FROM person IS A PARENT of the TO person
- reverse PARENT_OF means: the TO person IS A PARENT of the FROM person (FROM is the child)
- SPOUSE_OF means: the two people are married

STEP BY STEP — work through each step carefully:
1. Start at Person A
2. For each step, track who you are relative to the current node
3. Use the gender of each node to pick the right term

GENDER TERMS:
- Male parent = father, Female parent = mother
- Male child = son, Female child = daughter
- Male sibling = brother, Female sibling = sister
- Male grandparent = grandfather, Female grandparent = grandmother
- Male aunt/uncle = uncle, Female = aunt
- Male niece/nephew = nephew, Female = niece
- Male spouse = husband, Female spouse = wife

CRITICAL RULE:
Use the ACTUAL NAMES and GENDERS from the path steps.
Never assume father's side or mother's side — derive it from the actual intermediate nodes.

Give a warm, clear answer using actual names from the path.
Example: "Padmavati S is Sharan's aunt on his mother's side — she is the sister of his mother Hemavathi Veerapuram."
"""

# --- LLM ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,
)


def generate_cypher(state: AgentState) -> AgentState:
    log.info(f"Generating Cypher for: {state['question']}")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {state['question']}\n\nWrite the Cypher query:")
    ]
    response = llm.invoke(messages)
    cypher   = response.content.strip().replace("```cypher", "").replace("```", "").strip()
    is_rel   = cypher == "RELATIONSHIP_QUERY"
    log.info(f"Cypher: {cypher} | relation_query: {is_rel}")
    return {**state, "cypher": cypher, "is_relation_query": is_rel}


def execute_cypher(state: AgentState) -> AgentState:
    if state.get("is_relation_query"):
        log.info("Relationship query — extracting names...")
        extract_msg = [
            SystemMessage(content="""Extract exactly two full names from the question.
Return them as:
PERSON_A: <name>
PERSON_B: <name>
Nothing else."""),
            HumanMessage(content=state["question"])
        ]
        response = llm.invoke(extract_msg)
        lines    = response.content.strip().split("\n")
        person_a = lines[0].replace("PERSON_A:", "").strip()
        person_b = lines[1].replace("PERSON_B:", "").strip()
        log.info(f"Extracted names: '{person_a}' and '{person_b}'")
        path_data = get_path_between(person_a, person_b)
        log.info(f"Path data: {path_data}")
        return {**state, "path_data": path_data, "db_result": []}
    else:
        log.info(f"Executing Cypher: {state['cypher']}")
        result = run_cypher(state["cypher"])
        log.info(f"DB result: {result}")
        return {**state, "db_result": result, "path_data": {}}


def generate_answer(state: AgentState) -> AgentState:
    log.info("Generating answer...")
    error = None

    if state.get("is_relation_query"):
        path_data = state.get("path_data", {})
        if not path_data.get("found"):
            answer = "I couldn't find a connection between those two people. Please check the names are correct."
            log_query(
                question=state["question"], cypher=state["cypher"],
                db_result=[], answer=answer,
                is_relation_query=True, path_data=path_data,
                error="Path not found"
            )
            return {**state, "answer": answer}
        messages = [
            SystemMessage(content=RELATIONSHIP_INTERPRETER_PROMPT),
            HumanMessage(content=f"""
Question: {state['question']}
Person A: {path_data['person_a']}
Person B: {path_data['person_b']}
Path steps:
{chr(10).join([
    f"Step {i+1}: {s['from']} ({s['from_gender']}) --[{s['edge']} {s['direction']}]--> {s['to']} ({s['to_gender']})"
    for i, s in enumerate(path_data['steps'])
])}
Full path: {' -> '.join(path_data['nodes'])}
Interpret the relationship of {path_data['person_b']} to {path_data['person_a']}.
""")
        ]
    else:
        messages = [
            SystemMessage(content="You are a helpful family tree assistant. Answer naturally based on results. Mention actual names. Be warm. If empty say no results found."),
            HumanMessage(content=f"Question: {state['question']}\nCypher: {state['cypher']}\nResults: {state['db_result']}\nAnswer naturally.")
        ]

    try:
        response = llm.invoke(messages)
        answer   = response.content.strip()
    except Exception as e:
        answer = f"Sorry, I encountered an error generating the answer: {e}"
        error  = str(e)

    log.info(f"Answer: {answer}")

    # --- Write structured query log ---
    log_query(
        question=state["question"],
        cypher=state["cypher"],
        db_result=state.get("db_result", []),
        answer=answer,
        is_relation_query=state.get("is_relation_query", False),
        path_data=state.get("path_data", {}),
        error=error
    )

    return {**state, "answer": answer}


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("generate_cypher", generate_cypher)
    graph.add_node("execute_cypher",  execute_cypher)
    graph.add_node("generate_answer", generate_answer)
    graph.add_edge(START,             "generate_cypher")
    graph.add_edge("generate_cypher", "execute_cypher")
    graph.add_edge("execute_cypher",  "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


agent = build_agent()


def ask(question: str) -> str:
    result = agent.invoke({
        "question":          question,
        "messages":          [],
        "cypher":            "",
        "db_result":         [],
        "path_data":         {},
        "answer":            "",
        "is_relation_query": False,
    })
    return result["answer"]


if __name__ == "__main__":
    print("\n🌳 Family Tree AI Agent")
    print("=" * 40)
    print(f"📝 Logs → {LOG_FILE}")
    print(f"📋 Query log → {QUERY_LOG_FILE}")
    print("Type your question or 'quit' to exit\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        print(f"\nAgent: {ask(question)}\n")