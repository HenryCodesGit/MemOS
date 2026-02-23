"""
Server API Router for MemOS (Class-based handlers version).

This router demonstrates the improved architecture using class-based handlers
with dependency injection, providing better modularity and maintainability.

Comparison with function-based approach:
- Cleaner code: No need to pass dependencies in every endpoint
- Better testability: Easy to mock handler dependencies
- Improved extensibility: Add new handlers or modify existing ones easily
- Clear separation of concerns: Router focuses on routing, handlers handle business logic
"""

import os
import random as _random
import socket

from fastapi import APIRouter, HTTPException, Query

from memos.api import handlers
from memos.api.handlers.add_handler import AddHandler
from memos.api.handlers.base_handler import HandlerDependencies
from memos.api.handlers.chat_handler import ChatHandler
from memos.api.handlers.feedback_handler import FeedbackHandler
from memos.api.handlers.search_handler import SearchHandler
from memos.api.product_models import (
    AllStatusResponse,
    APIADDRequest,
    APIChatCompleteRequest,
    APIFeedbackRequest,
    APISearchRequest,
    ChatBusinessRequest,
    ChatPlaygroundRequest,
    ChatRequest,
    DeleteMemoryByRecordIdRequest,
    DeleteMemoryByRecordIdResponse,
    DeleteMemoryRequest,
    DeleteMemoryResponse,
    ExistMemCubeIdRequest,
    ExistMemCubeIdResponse,
    GetMemoryDashboardRequest,
    GetMemoryPlaygroundRequest,
    GetMemoryRequest,
    GetMemoryResponse,
    GetUserNamesByMemoryIdsRequest,
    GetUserNamesByMemoryIdsResponse,
    GraphChildrenRequest,
    GraphTopicsRequest,
    MemoryResponse,
    RecoverMemoryByRecordIdRequest,
    RecoverMemoryByRecordIdResponse,
    SearchResponse,
    StatusResponse,
    SuggestionRequest,
    SuggestionResponse,
    TaskQueueResponse,
)
from memos.log import get_logger
from memos.mem_scheduler.base_scheduler import BaseScheduler
from memos.mem_scheduler.utils.status_tracker import TaskStatusTracker


logger = get_logger(__name__)

router = APIRouter(prefix="/product", tags=["Server API"])

# Instance ID for identifying this server instance in logs and responses
INSTANCE_ID = f"{socket.gethostname()}:{os.getpid()}:{_random.randint(1000, 9999)}"

# Initialize all server components
components = handlers.init_server()

# Create dependency container
dependencies = HandlerDependencies.from_init_server(components)

# Initialize all handlers with dependency injection
search_handler = SearchHandler(dependencies)
add_handler = AddHandler(dependencies)
chat_handler = (
    ChatHandler(
        dependencies,
        components["chat_llms"],
        search_handler,
        add_handler,
        online_bot=components.get("online_bot"),
    )
    if os.getenv("ENABLE_CHAT_API", "false") == "true"
    else None
)
feedback_handler = FeedbackHandler(dependencies)
# Extract commonly used components for function-based handlers
# (These can be accessed from the components dict without unpacking all of them)
mem_scheduler: BaseScheduler = components["mem_scheduler"]
llm = components["llm"]
naive_mem_cube = components["naive_mem_cube"]
redis_client = components["redis_client"]
status_tracker = TaskStatusTracker(redis_client=redis_client)
graph_db = components["graph_db"]
vector_db = components["vector_db"]


# =============================================================================
# Graph Browsing API Endpoints
# =============================================================================


@router.post("/graph/topics", summary="Browse memory topic nodes")
def graph_topics(req: GraphTopicsRequest):
    """Return all topic nodes with child counts, unlinked count, and total leaf count."""
    try:
        driver = graph_db.driver

        with driver.session() as session:
            # Topic nodes with child counts
            topics_result = session.run(
                """
                MATCH (n:Memory)
                WHERE n.type = 'topic'
                  AND n.user_name = $user_name
                  AND n.status = 'activated'
                OPTIONAL MATCH (n)-[:PARENT]->(c:Memory)
                WHERE c.user_name = $user_name
                RETURN n.id AS id, n.key AS key, n.memory AS memory,
                       n.tags AS tags, count(c) AS child_count
                ORDER BY count(c) DESC
                """,
                user_name=req.user_name,
            )
            topics = [
                {
                    "id": r["id"],
                    "key": r["key"],
                    "memory": r["memory"],
                    "tags": r["tags"],
                    "child_count": r["child_count"],
                }
                for r in topics_result
            ]

            # Unlinked nodes (non-topic, no incoming PARENT edge)
            unlinked_result = session.run(
                """
                MATCH (n:Memory)
                WHERE n.user_name = $user_name
                  AND n.status = 'activated'
                  AND (n.type IS NULL OR n.type <> 'topic')
                  AND NOT (()-[:PARENT]->(n))
                RETURN count(n) AS cnt
                """,
                user_name=req.user_name,
            )
            unlinked_count = unlinked_result.single()["cnt"]

            # Total leaf memories (non-topic)
            total_result = session.run(
                """
                MATCH (n:Memory)
                WHERE n.user_name = $user_name
                  AND n.status = 'activated'
                  AND (n.type IS NULL OR n.type <> 'topic')
                RETURN count(n) AS cnt
                """,
                user_name=req.user_name,
            )
            total_memories = total_result.single()["cnt"]

        return {
            "topics": topics,
            "unlinked_count": unlinked_count,
            "total_memories": total_memories,
        }
    except Exception as e:
        logger.error(f"graph/topics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/children", summary="Browse children of a topic node")
def graph_children(req: GraphChildrenRequest):
    """Return children of a topic node, or unlinked nodes if parent_id is null."""
    try:
        driver = graph_db.driver

        with driver.session() as session:
            if req.parent_id is not None:
                # Get parent info
                parent_result = session.run(
                    """
                    MATCH (n:Memory)
                    WHERE n.id = $parent_id
                      AND n.user_name = $user_name
                    RETURN n.id AS id, n.key AS key, n.memory AS memory
                    """,
                    parent_id=req.parent_id,
                    user_name=req.user_name,
                )
                parent_record = parent_result.single()
                parent = (
                    {
                        "id": parent_record["id"],
                        "key": parent_record["key"],
                        "memory": parent_record["memory"],
                    }
                    if parent_record
                    else None
                )

                # Get children
                children_result = session.run(
                    """
                    MATCH (p:Memory)-[:PARENT]->(c:Memory)
                    WHERE p.id = $parent_id
                      AND c.user_name = $user_name
                      AND c.status = 'activated'
                    RETURN c.id AS id, c.memory AS memory, c.key AS key,
                           c.tags AS tags, c.updated_at AS updated_at, c.status AS status
                    """,
                    parent_id=req.parent_id,
                    user_name=req.user_name,
                )
            else:
                parent = None
                # Unlinked nodes
                children_result = session.run(
                    """
                    MATCH (n:Memory)
                    WHERE n.user_name = $user_name
                      AND n.status = 'activated'
                      AND (n.type IS NULL OR n.type <> 'topic')
                      AND NOT (()-[:PARENT]->(n))
                    RETURN n.id AS id, n.memory AS memory, n.key AS key,
                           n.tags AS tags, n.updated_at AS updated_at, n.status AS status
                    """,
                    user_name=req.user_name,
                )

            children = [
                {
                    "id": r["id"],
                    "memory": r["memory"],
                    "key": r["key"],
                    "tags": r["tags"],
                    "updated_at": r["updated_at"],
                    "status": r["status"],
                }
                for r in children_result
            ]

        return {
            "parent": parent,
            "children": children,
        }
    except Exception as e:
        logger.error(f"graph/children error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Search API Endpoints
# =============================================================================


@router.post("/search", summary="Search memories", response_model=SearchResponse)
def search_memories(search_req: APISearchRequest):
    """
    Search memories for a specific user.

    This endpoint uses the class-based SearchHandler for better code organization.
    """
    search_results = search_handler.handle_search_memories(search_req)
    return search_results


# =============================================================================
# Add API Endpoints
# =============================================================================


@router.post("/add", summary="Add memories", response_model=MemoryResponse)
def add_memories(add_req: APIADDRequest):
    """
    Add memories for a specific user.

    This endpoint uses the class-based AddHandler for better code organization.
    """
    return add_handler.handle_add_memories(add_req)


# =============================================================================
# Scheduler API Endpoints
# =============================================================================


@router.get(  # Changed from post to get
    "/scheduler/allstatus",
    summary="Get detailed scheduler status",
    response_model=AllStatusResponse,
)
def scheduler_allstatus():
    """Get detailed scheduler status including running tasks and queue metrics."""
    return handlers.scheduler_handler.handle_scheduler_allstatus(
        mem_scheduler=mem_scheduler, status_tracker=status_tracker
    )


@router.get(  # Changed from post to get
    "/scheduler/status", summary="Get scheduler running status", response_model=StatusResponse
)
def scheduler_status(
    user_id: str = Query(..., description="User ID"),
    task_id: str | None = Query(None, description="Optional Task ID to query a specific task"),
):
    """Get scheduler running status."""
    return handlers.scheduler_handler.handle_scheduler_status(
        user_id=user_id,
        task_id=task_id,
        status_tracker=status_tracker,
    )


@router.get(  # Changed from post to get
    "/scheduler/task_queue_status",
    summary="Get scheduler task queue status",
    response_model=TaskQueueResponse,
)
def scheduler_task_queue_status(
    user_id: str = Query(..., description="User ID whose queue status is requested"),
):
    """Get scheduler task queue backlog/pending status for a user."""
    return handlers.scheduler_handler.handle_task_queue_status(
        user_id=user_id, mem_scheduler=mem_scheduler
    )


@router.post("/scheduler/wait", summary="Wait until scheduler is idle for a specific user")
def scheduler_wait(
    user_name: str,
    timeout_seconds: float = 120.0,
    poll_interval: float = 0.5,
):
    """Wait until scheduler is idle for a specific user."""
    return handlers.scheduler_handler.handle_scheduler_wait(
        user_name=user_name,
        status_tracker=status_tracker,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
    )


@router.get("/scheduler/wait/stream", summary="Stream scheduler progress for a user")
def scheduler_wait_stream(
    user_name: str,
    timeout_seconds: float = 120.0,
    poll_interval: float = 0.5,
):
    """Stream scheduler progress via Server-Sent Events (SSE)."""
    return handlers.scheduler_handler.handle_scheduler_wait_stream(
        user_name=user_name,
        status_tracker=status_tracker,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        instance_id=INSTANCE_ID,
    )


# =============================================================================
# Chat API Endpoints
# =============================================================================


@router.post("/chat/complete", summary="Chat with MemOS (Complete Response)")
def chat_complete(chat_req: APIChatCompleteRequest):
    """
    Chat with MemOS for a specific user. Returns complete response (non-streaming).

    This endpoint uses the class-based ChatHandler.
    """
    if chat_handler is None:
        raise HTTPException(
            status_code=503, detail="Chat service is not available. Chat handler not initialized."
        )
    return chat_handler.handle_chat_complete(chat_req)


@router.post("/chat/stream", summary="Chat with MemOS")
def chat_stream(chat_req: ChatRequest):
    """
    Chat with MemOS for a specific user. Returns SSE stream.

    This endpoint uses the class-based ChatHandler which internally
    composes SearchHandler and AddHandler for a clean architecture.
    """
    if chat_handler is None:
        raise HTTPException(
            status_code=503, detail="Chat service is not available. Chat handler not initialized."
        )
    return chat_handler.handle_chat_stream(chat_req)


@router.post("/chat/stream/playground", summary="Chat with MemOS playground")
def chat_stream_playground(chat_req: ChatPlaygroundRequest):
    """
    Chat with MemOS for a specific user. Returns SSE stream.

    This endpoint uses the class-based ChatHandler which internally
    composes SearchHandler and AddHandler for a clean architecture.
    """
    if chat_handler is None:
        raise HTTPException(
            status_code=503, detail="Chat service is not available. Chat handler not initialized."
        )
    return chat_handler.handle_chat_stream_playground(chat_req)


# =============================================================================
# Suggestion API Endpoints
# =============================================================================


@router.post(
    "/suggestions",
    summary="Get suggestion queries",
    response_model=SuggestionResponse,
)
def get_suggestion_queries(suggestion_req: SuggestionRequest):
    """Get suggestion queries for a specific user with language preference."""
    return handlers.suggestion_handler.handle_get_suggestion_queries(
        user_id=suggestion_req.mem_cube_id,
        language=suggestion_req.language,
        message=suggestion_req.message,
        llm=llm,
        naive_mem_cube=naive_mem_cube,
    )


# =============================================================================
# Memory Retrieval Delete API Endpoints
# =============================================================================


@router.post("/get_all", summary="Get all memories for user", response_model=MemoryResponse)
def get_all_memories(memory_req: GetMemoryPlaygroundRequest):
    """
    Get all memories or subgraph for a specific user.

    If search_query is provided, returns a subgraph based on the query.
    Otherwise, returns all memories of the specified type.
    """
    if memory_req.search_query:
        return handlers.memory_handler.handle_get_subgraph(
            user_id=memory_req.user_id,
            mem_cube_id=(
                memory_req.mem_cube_ids[0] if memory_req.mem_cube_ids else memory_req.user_id
            ),
            query=memory_req.search_query,
            top_k=200,
            naive_mem_cube=naive_mem_cube,
            search_type=memory_req.search_type,
        )
    else:
        return handlers.memory_handler.handle_get_all_memories(
            user_id=memory_req.user_id,
            mem_cube_id=(
                memory_req.mem_cube_ids[0] if memory_req.mem_cube_ids else memory_req.user_id
            ),
            memory_type=memory_req.memory_type or "text_mem",
            naive_mem_cube=naive_mem_cube,
        )


@router.post("/get_memory", summary="Get memories for user", response_model=GetMemoryResponse)
def get_memories(memory_req: GetMemoryRequest):
    return handlers.memory_handler.handle_get_memories(
        get_mem_req=memory_req,
        naive_mem_cube=naive_mem_cube,
    )


@router.get("/get_memory/{memory_id}", summary="Get memory by id", response_model=GetMemoryResponse)
def get_memory_by_id(memory_id: str):
    return handlers.memory_handler.handle_get_memory(
        memory_id=memory_id,
        naive_mem_cube=naive_mem_cube,
    )


@router.post("/get_memory_by_ids", summary="Get memory by ids", response_model=GetMemoryResponse)
def get_memory_by_ids(memory_ids: list[str]):
    return handlers.memory_handler.handle_get_memory_by_ids(
        memory_ids=memory_ids,
        naive_mem_cube=naive_mem_cube,
    )


@router.post(
    "/delete_memory", summary="Delete memories for user", response_model=DeleteMemoryResponse
)
def delete_memories(memory_req: DeleteMemoryRequest):
    return handlers.memory_handler.handle_delete_memories(
        delete_mem_req=memory_req, naive_mem_cube=naive_mem_cube
    )


# =============================================================================
# Feedback API Endpoints
# =============================================================================


@router.post("/feedback", summary="Feedback memories", response_model=MemoryResponse)
def feedback_memories(feedback_req: APIFeedbackRequest):
    """
    Feedback memories for a specific user.

    This endpoint uses the class-based FeedbackHandler for better code organization.
    """
    return feedback_handler.handle_feedback_memories(feedback_req)


# =============================================================================
# Other API Endpoints (for internal use)
# =============================================================================


@router.post(
    "/get_user_names_by_memory_ids",
    summary="Get user names by memory ids",
    response_model=GetUserNamesByMemoryIdsResponse,
)
def get_user_names_by_memory_ids(request: GetUserNamesByMemoryIdsRequest):
    """Get user names by memory ids."""
    result = graph_db.get_user_names_by_memory_ids(memory_ids=request.memory_ids)
    if vector_db:
        prefs = []
        for collection_name in ["explicit_preference", "implicit_preference"]:
            prefs.extend(
                vector_db.get_by_ids(collection_name=collection_name, ids=request.memory_ids)
            )
        result.update({pref.id: pref.payload.get("mem_cube_id", None) for pref in prefs})
    return GetUserNamesByMemoryIdsResponse(
        code=200,
        message="Successfully",
        data=result,
    )


@router.post(
    "/exist_mem_cube_id",
    summary="Check if mem cube id exists",
    response_model=ExistMemCubeIdResponse,
)
def exist_mem_cube_id(request: ExistMemCubeIdRequest):
    """(inner) Check if mem cube id exists."""
    return ExistMemCubeIdResponse(
        code=200,
        message="Successfully",
        data=graph_db.exist_user_name(user_name=request.mem_cube_id),
    )


@router.post("/chat/stream/business_user", summary="Chat with MemOS for business user")
def chat_stream_business_user(chat_req: ChatBusinessRequest):
    """(inner) Chat with MemOS for a specific business user. Returns SSE stream."""
    if chat_handler is None:
        raise HTTPException(
            status_code=503, detail="Chat service is not available. Chat handler not initialized."
        )

    return chat_handler.handle_chat_stream_for_business_user(chat_req)


@router.post(
    "/delete_memory_by_record_id",
    summary="Delete memory by record id",
    response_model=DeleteMemoryByRecordIdResponse,
)
def delete_memory_by_record_id(memory_req: DeleteMemoryByRecordIdRequest):
    """(inner) Delete memory nodes by mem_cube_id (user_name) and delete_record_id. Record id is inner field, just for delete and recover memory, not for user to set."""
    graph_db.delete_node_by_mem_cube_id(
        mem_cube_id=memory_req.mem_cube_id,
        delete_record_id=memory_req.record_id,
        hard_delete=memory_req.hard_delete,
    )

    return DeleteMemoryByRecordIdResponse(
        code=200,
        message="Called Successfully",
        data={"status": "success"},
    )


@router.post(
    "/recover_memory_by_record_id",
    summary="Recover memory by record id",
    response_model=RecoverMemoryByRecordIdResponse,
)
def recover_memory_by_record_id(memory_req: RecoverMemoryByRecordIdRequest):
    """(inner) Recover memory nodes by mem_cube_id (user_name) and delete_record_id. Record id is inner field, just for delete and recover memory, not for user to set."""
    graph_db.recover_memory_by_mem_cube_id(
        mem_cube_id=memory_req.mem_cube_id,
        delete_record_id=memory_req.delete_record_id,
    )

    return RecoverMemoryByRecordIdResponse(
        code=200,
        message="Called Successfully",
        data={"status": "success"},
    )


@router.post(
    "/get_memory_dashboard", summary="Get memories for dashboard", response_model=GetMemoryResponse
)
def get_memories_dashboard(memory_req: GetMemoryDashboardRequest):
    return handlers.memory_handler.handle_get_memories_dashboard(
        get_mem_req=memory_req,
        naive_mem_cube=naive_mem_cube,
    )
