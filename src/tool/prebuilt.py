from pydantic import BaseModel, Field
from src.tool import tool
from src.tool.rag import RAGTool
from src.memory.manager import MemoryManager

# ─── Singleton RAG instance ───
_rag_tool: RAGTool = None


def get_rag_tool(knowledge_base_path: str = "./rag_kb") -> RAGTool:
    """Get or create the global RAG tool instance."""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = RAGTool(knowledge_base_path=knowledge_base_path)
    return _rag_tool


# ─── Singleton MemoryManager instance ───
_memory_manager: MemoryManager = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global MemoryManager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(storage_path="./memory_data")
        # Share embedding function from RAG tool
        rag = get_rag_tool()
        _memory_manager.set_embed_fn(rag._embed)
    return _memory_manager


# ═══════════════════════════════════════════════════════════
# User Interaction
# ═══════════════════════════════════════════════════════════

class UserInteraction(BaseModel):
    question: str = Field(..., description="The question you want to ask the user.",
                          example="Hello, how are you?")


@tool('User Interaction Tool', args_schema=UserInteraction)
def user_interface_tool(question: str):
    '''
    The best way to interact with the user for clarification or asking questions
    and taking inputs from the user.
    '''
    ai_query = f'AI: {question}\nUser: '
    user_message = input(ai_query)
    return user_message


# ═══════════════════════════════════════════════════════════
# RAG Tools
# ═══════════════════════════════════════════════════════════

class AddDocumentInput(BaseModel):
    text: str = Field(..., description="The full text content to add to the knowledge base.")
    doc_id: str = Field(default="", description="Optional document ID. Leave empty for auto-generation.")
    title: str = Field(default="", description="Optional title for the document.")


@tool('RAG Add Document Tool', args_schema=AddDocumentInput)
def rag_add_document(text: str, doc_id: str = "", title: str = ""):
    '''
    Add a text document to the RAG knowledge base.
    The document will be split into chunks, embedded, and stored in the vector database
    for later semantic retrieval. Use this when you need to remember important information
    for future queries.
    '''
    rag = get_rag_tool()
    metadata = {"title": title} if title else {}
    result = rag.add_text(text, doc_id=doc_id, metadata=metadata)
    if result.get("status") == "ok":
        return f"Document '{result['doc_id']}' added successfully with {result['chunk_count']} chunks."
    return f"Error: {result.get('message', 'Unknown error')}"


class SearchInput(BaseModel):
    query: str = Field(..., description="The search query to find relevant documents.")
    top_k: int = Field(default=3, description="Number of top results to return (max 10).")


@tool('RAG Search Tool', args_schema=SearchInput)
def rag_search(query: str, top_k: int = 3):
    '''
    Search the RAG knowledge base for documents semantically similar to the query.
    Returns the most relevant text chunks with similarity scores.
    Use this when you need to recall previously stored information.
    '''
    rag = get_rag_tool()
    results = rag.search(query, top_k=min(top_k, 10))
    if not results:
        return "No relevant documents found in the knowledge base."
    parts = []
    for i, r in enumerate(results):
        parts.append(f"[{i + 1}] Score: {r['score']:.3f}\n{r['content'][:500]}")
    return "\n\n".join(parts) if parts else "No results."


class AskInput(BaseModel):
    query: str = Field(..., description="The question to retrieve context for.")
    top_k: int = Field(default=5, description="Number of context snippets to retrieve.")


@tool('RAG Ask Tool', args_schema=AskInput)
def rag_ask(query: str, top_k: int = 5):
    '''
    Retrieve context snippets from the knowledge base to help answer a question.
    Returns formatted context that can be used to generate an informed answer.
    Use this BEFORE answering a question to check if relevant information exists.
    '''
    rag = get_rag_tool()
    context = rag.ask(query, top_k=min(top_k, 10))
    stats = rag.stats()
    header = f"[Knowledge Base: {stats.get('chunks', 0)} chunks total]\n\n"
    return header + context


# ═══════════════════════════════════════════════════════════
# Memory Tools — Working / Episodic / Semantic
# ═══════════════════════════════════════════════════════════

class RememberInput(BaseModel):
    content: str = Field(..., description="What to remember (text content).")
    memory_type: str = Field(default="working",
                             description="Memory type: 'working' (temporary), 'episodic' (event), or 'semantic' (knowledge).")
    importance: float = Field(default=0.5,
                              description="Importance 0.0–1.0. Higher = kept longer, ranked higher in search.")


@tool('Memory Remember Tool', args_schema=RememberInput)
def memory_remember(content: str, memory_type: str = "working",
                    importance: float = 0.5):
    '''
    Store information into the agent's memory system.

    - Use memory_type='working' for temporary info (session-only, auto-expires).
    - Use memory_type='episodic' for events, conversations, experiences.
    - Use memory_type='semantic' for facts, concepts, definitions, knowledge.

    The memory system automatically retrieves relevant memories when you call memory_recall.
    '''
    mm = get_memory_manager()
    item_id = mm.add(content, memory_type=memory_type, importance=importance)
    stats = mm.stats()
    return (f"Stored in {memory_type} memory (id: {item_id}, importance: {importance}).\n"
            f"Working: {stats['working']['count']} | Episodic: {stats['episodic']['count']} | Semantic: {stats['semantic']['concepts']} concepts")


class RecallInput(BaseModel):
    query: str = Field(..., description="What to search for in memory.")
    memory_types: str = Field(default="all",
                              description="Which memory types to search. 'all', or comma-separated: 'working,episodic,semantic'.")
    limit: int = Field(default=5, description="Max results per memory type.")


@tool('Memory Recall Tool', args_schema=RecallInput)
def memory_recall(query: str, memory_types: str = "all", limit: int = 5):
    '''
    Search ALL memory systems for relevant information.
    Returns results from working memory, episodic memory, and semantic memory.

    Use this BEFORE answering any question — the agent should always check
    if it already knows relevant information from past interactions.
    '''
    mm = get_memory_manager()
    types = None if memory_types == "all" else [t.strip() for t in memory_types.split(",")]
    results = mm.search(query, memory_types=types, limit=limit)

    total = sum(len(v) for v in results.values())
    if total == 0:
        return "No relevant memories found. Try storing some information first with memory_remember."

    parts = []
    for mem_type, items in results.items():
        if not items:
            continue
        parts.append(f"--- {mem_type.upper()} MEMORY ---")
        for item in items:
            score_str = f" [score: {item.get('score', 'N/A')}]" if 'score' in item else ""
            content = item.get('content', str(item))[:300]
            parts.append(f"{score_str} {content}")
    return "\n\n".join(parts)


class LearnInput(BaseModel):
    concept: str = Field(..., description="The concept name or topic.")
    definition: str = Field(..., description="The definition or explanation of the concept.")
    domain: str = Field(default="", description="Knowledge domain (e.g. 'programming', 'history').")
    importance: float = Field(default=0.7, description="Importance 0.0–1.0.")


@tool('Memory Learn Tool', args_schema=LearnInput)
def memory_learn(concept: str, definition: str, domain: str = "",
                 importance: float = 0.7):
    '''
    Learn a new concept or fact — stores it into semantic (long-term) memory.
    Use this when the user teaches you something new that you should remember permanently.
    '''
    mm = get_memory_manager()
    cid = mm.add_knowledge(concept, definition, domain, importance)
    return f"Learned concept '{concept}' (id: {cid}, domain: {domain or 'general'}). Total concepts: {mm.semantic.stats()['concepts']}"


class RecordEventInput(BaseModel):
    content: str = Field(..., description="Description of the event or experience.")
    event_type: str = Field(default="general", description="Event category: 'conversation', 'task', 'observation', etc.")
    importance: float = Field(default=0.5, description="Importance 0.0–1.0.")


@tool('Memory Record Event Tool', args_schema=RecordEventInput)
def memory_record_event(content: str, event_type: str = "general",
                        importance: float = 0.5):
    '''
    Record an event or experience into episodic memory.
    Events are stored with timestamps and can be retrieved in chronological order.
    Use this for logging user interactions, task completions, or notable occurrences.
    '''
    mm = get_memory_manager()
    eid = mm.episodic.add(content, importance=importance,
                          event_type=event_type, session_id=mm.session_id)
    return f"Event recorded (id: {eid}, type: {event_type}). Episodic memory now has {mm.episodic.stats()['count']} events."
