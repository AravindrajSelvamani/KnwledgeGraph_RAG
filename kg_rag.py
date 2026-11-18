"""
=============================================================================
 Tamil Nadu Government Schemes - Knowledge Graph RAG
=============================================================================

 WHAT THIS DOES:
1. Reads FAQ.txt using LangChain TextLoader & splits into chunks
2. Uses OpenAI + LLMGraphTransformer to extract Entities & Relationships
3. Stores the Knowledge Graph in Neo4j Aura DB
4. Translates user questions into Cypher queries + generates answers via OpenAI

 ARCHITECTURE (Simple View):

   FAQ.txt --> [TextLoader] --> [LLMGraphTransformer] --> [Neo4j Aura DB]
                                (Entity & Relationship     (nodes &
                                 Extraction)                relationships)
                                                                 |
   Answer  <-- [OpenAI LLM] <-- [GraphCypherQAChain] <-----------+
                                (Generates Cypher &
                                 searches graph)

 GRAPH STRUCTURE (Auto-extracted by LLM):
   (Scheme) --[:MANAGED_BY]--> (Department)
   (Scheme) --[:PROVIDES_BENEFIT]--> (Benefit)
   (Scheme) --[:REQUIRES_DOCUMENT]--> (Document)
   (Scheme) --[:APPLY_VIA]--> (Portal)

=============================================================================
"""

import os
import sys
import certifi

# Fix SSL certificate verification on Windows
# Must run BEFORE importing Neo4j / LangChain modules
if sys.platform == "win32":
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, LLMGraphTransformer, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate


# ─────────────────────────────────────────────────
# STEP 1: Load environment variables from .env file
# ─────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Quick check: make sure all keys are set
for var_name in ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]:
    if not os.getenv(var_name):
        raise ValueError(f"Missing {var_name} in .env file. See .env.example")


# ─────────────────────────────────────────────────
# STEP 2: Load and split FAQ.txt (DocumentLoader)
# ─────────────────────────────────────────────────
# We use LangChain's TextLoader to read FAQ.txt and
# RecursiveCharacterTextSplitter to keep each Q&A block intact.

def load_and_split_faq(file_path: str):
    """
    Reads FAQ.txt using LangChain TextLoader and returns Document chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FAQ file not found: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\nQ: ", "\n\n", "\n", " ", ""]
    )
    documents = splitter.split_documents(raw_documents)
    print(f"  Loaded and split {file_path} into {len(documents)} chunks.")
    return documents


# ─────────────────────────────────────────────────
# STEP 3: Extract Entities & Push to Neo4j Aura DB
# ─────────────────────────────────────────────────
# Instead of manual string splitting, LLMGraphTransformer uses OpenAI
# to automatically identify Schemes, Departments, Benefits, Documents, etc.
# and connect them with meaningful relationships.

def clear_graph(graph: Neo4jGraph):
    """Delete all existing nodes and relationships (clean start)."""
    graph.query("MATCH (n) DETACH DELETE n")
    print("  Cleared existing graph data.")


def extract_and_push_graph(graph: Neo4jGraph, documents, llm):
    """
    Extracts entities & relationships via LLMGraphTransformer
    and saves them into Neo4j Aura DB.
    """
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[
            "Scheme",
            "Department",
            "Benefit",
            "Eligibility",
            "Document",
            "Portal",
            "TargetGroup",
        ],
        allowed_relationships=[
            "MANAGED_BY",
            "PROVIDES_BENEFIT",
            "ELIGIBLE_FOR",
            "REQUIRES_DOCUMENT",
            "APPLY_VIA",
            "TARGETS",
        ],
        node_properties=["description"],
    )

    print("  Extracting entities and relationships using OpenAI...")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Preview extracted nodes & relationships from the first chunk (for learning)
    if graph_documents:
        print(f"  Sample Nodes from Chunk 1: {[f'{n.id} ({n.type})' for n in graph_documents[0].nodes]}")
        print(f"  Sample Relations from Chunk 1: {[f'{r.source.id} -[{r.type}]-> {r.target.id}' for r in graph_documents[0].relationships]}")

    # Push to Neo4j Aura DB
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True,
    )
    graph.refresh_schema()
    print(f"  Pushed {len(graph_documents)} graph documents to Neo4j Knowledge Graph!")


# ─────────────────────────────────────────────────
# STEP 4: Create Graph Search + OpenAI RAG Chain
# ─────────────────────────────────────────────────
# GraphCypherQAChain reads your Neo4j schema, writes a Cypher query
# for the user's question, runs it on Aura DB, and uses OpenAI to
# write a clean, natural answer grounded in the graph results.

def create_rag_chain(graph: Neo4jGraph, llm):
    """
    Creates LangChain's GraphCypherQAChain with a custom QA prompt.
    """
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful Tamil Nadu Government Schemes assistant that answers questions
using ONLY the context retrieved from the Knowledge Graph.

=== Knowledge Graph Context ===
{context}
================================

User Question: {question}

Instructions:
- Answer based ONLY on the Knowledge Graph context above.
- Do not invent scheme names, benefits, eligibility rules, or documents.
- If the context is empty or doesn't contain relevant info, say:
  "I don't have information about that in my knowledge graph."
- Keep the answer clear, polite, and concise.

Answer:""",
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        qa_prompt=qa_prompt,
        verbose=True,                 # Prints the generated Cypher query so you can see how it searches
        allow_dangerous_requests=True # Required safety flag for executing Cypher queries
    )
    return chain


def ask_question(chain, user_question: str) -> str:
    """
    Full Graph RAG pipeline:
    1. LLM translates user_question -> Cypher query
    2. Runs Cypher query against Neo4j Aura DB
    3. Passes graph results + question -> OpenAI LLM
    4. Returns final grounded answer
    """
    response = chain.invoke({"query": user_question})
    return response["result"]


# ─────────────────────────────────────────────────
# STEP 5: Main -- tie it all together
# ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Tamil Nadu Government Schemes - Knowledge Graph RAG")
    print("  (LangChain + LLMGraphTransformer + Neo4j + OpenAI)")
    print("=" * 60)

    # --- Connect to Neo4j ---
    print("\n[1/4] Connecting to Neo4j Aura DB...")
    print(f"  URI: {NEO4J_URI}")
    if "xxxxxxxx" in NEO4J_URI:
        print("\n  ERROR: You still have the placeholder URI in your .env file!")
        print("  Go to https://console.neo4j.io, click your instance,")
        print("  and copy the 'Connection URI' into NEO4J_URI in .env")
        return

    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            enhanced_schema=True,
        )
    except Exception as e:
        print(f"\n  ERROR: Could not connect to Neo4j: {e}")
        print("\n  Checklist:")
        print("  1. Is your Neo4j Aura instance running? (check console.neo4j.io)")
        print("  2. Is NEO4J_URI correct? (try neo4j+ssc://... if behind a VPN/firewall)")
        print("  3. Is NEO4J_USERNAME correct? (usually 'neo4j' for Aura)")
        print("  4. Is NEO4J_PASSWORD correct?")
        return
    print("  Connected successfully!")

    # --- Initialize OpenAI LLM ---
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    # --- Load, Extract Entities, and Push FAQs ---
    print("\n[2/4] Loading FAQ.txt...")
    faq_file = os.path.join(os.path.dirname(__file__), "FAQ.txt")
    documents = load_and_split_faq(faq_file)

    print("\n[3/4] Building Knowledge Graph in Neo4j Aura DB...")
    # TIP: After running once, you can comment out clear_graph() and extract_and_push_graph()
    # to skip re-ingesting and start chatting immediately!
    clear_graph(graph)
    extract_and_push_graph(graph, documents, llm)

    # --- Create RAG chain ---
    print("\n[4/4] Setting up Graph Search RAG chain (LangChain + OpenAI)...")
    chain = create_rag_chain(graph, llm)
    print("  Ready!\n")

    # --- Interactive Q&A Loop ---
    print("=" * 60)
    print("  Ask me anything about Tamil Nadu Government Schemes!")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        print()
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        try:
            answer = ask_question(chain, user_input)
            print(f"\nAssistant: {answer}")
        except Exception as e:
            print(f"\n  Error answering question: {e}")


if __name__ == "__main__":
    main()