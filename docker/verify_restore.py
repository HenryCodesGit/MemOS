"""
Post-restore verification script for MemOS memory data.

Runs inside the memos container via:
    docker exec memos-api-docker python /app/docker/verify_restore.py /app/backups/<timestamp>

Checks:
  1. Neo4j node count matches manifest
  2. Neo4j edge count matches manifest
  3. Qdrant point count matches manifest
  4. Sample search returns results
  5. Embedding dimensions correct (spot-check = 1024)
  6. cube_id matches on all Qdrant points
  7. PARENT edges exist (reorganizer structure)
  8. Scheduler status
"""

import json
import os
import sys
import urllib.request

from memos.api.handlers.component_init import init_server
from memos.log import get_logger

logger = get_logger(__name__)

SEARCH_TEST_QUERIES = [
    "project architecture and setup",
    "debugging and troubleshooting",
    "user preferences",
]


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def run_verification(backup_dir: str) -> None:
    cube_id = os.environ.get("MOS_CUBE_ID")
    if not cube_id:
        print("ERROR: MOS_CUBE_ID environment variable is not set.")
        sys.exit(1)

    manifest_file = os.path.join(backup_dir, "backup_manifest.json")
    manifest = None
    if os.path.exists(manifest_file):
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        print(f"WARNING: No manifest found at {manifest_file}")
        print("Running checks without expected counts.\n")

    # --- Initialize ---
    print("Initializing server components...")
    components = init_server()
    naive = components["naive_mem_cube"]
    text_mem = naive.text_mem
    graph_store = text_mem.graph_store

    results = []
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60 + "\n")

    # --- Check 1: Neo4j node count ---
    try:
        data = graph_store.export_graph(
            include_embedding=False,
            user_name=cube_id,
        )
        actual_nodes = data.get("total_nodes", len(data.get("nodes", [])))
        actual_edges = data.get("total_edges", len(data.get("edges", [])))

        if manifest:
            expected_nodes = manifest.get("node_count", 0)
            results.append(check(
                "Neo4j node count",
                actual_nodes >= expected_nodes,
                f"expected={expected_nodes}, actual={actual_nodes}",
            ))
        else:
            results.append(check(
                "Neo4j node count",
                actual_nodes > 0,
                f"actual={actual_nodes}",
            ))
    except Exception as e:
        results.append(check("Neo4j node count", False, str(e)))
        actual_nodes = 0
        actual_edges = 0

    # --- Check 2: Neo4j edge count ---
    if manifest:
        expected_edges = manifest.get("edge_count", 0)
        results.append(check(
            "Neo4j edge count",
            actual_edges >= expected_edges,
            f"expected={expected_edges}, actual={actual_edges}",
        ))
    else:
        results.append(check(
            "Neo4j edge count",
            actual_edges >= 0,
            f"actual={actual_edges}",
        ))

    # --- Check 3: Qdrant point count ---
    qdrant_count = 0
    if hasattr(graph_store, "vec_db"):
        try:
            qdrant_count = graph_store.vec_db.count({"user_name": cube_id})
            if manifest:
                expected_qdrant = manifest.get("qdrant_point_count", 0)
                results.append(check(
                    "Qdrant point count",
                    qdrant_count >= expected_qdrant,
                    f"expected={expected_qdrant}, actual={qdrant_count}",
                ))
            else:
                results.append(check(
                    "Qdrant point count",
                    qdrant_count > 0,
                    f"actual={qdrant_count}",
                ))
        except Exception as e:
            results.append(check("Qdrant point count", False, str(e)))
    else:
        results.append(check("Qdrant point count", False, "no vec_db on graph_store"))

    # --- Check 4: Sample search returns results ---
    search_ok = False
    for query in SEARCH_TEST_QUERIES:
        try:
            req_data = json.dumps({
                "user_id": os.environ.get("MOS_USER_ID", "default_user"),
                "readable_cube_ids": [cube_id],
                "query": query,
            }).encode()
            req = urllib.request.Request(
                "http://localhost:8000/product/search",
                data=req_data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read())
                # Response may nest under "data" key
                data = resp_data.get("data", resp_data)
                text_results = data.get("text_mem", [])
                memories = []
                for group in text_results:
                    memories.extend(group.get("memories", []))
                if memories:
                    search_ok = True
                    break
        except Exception:
            continue

    results.append(check(
        "Sample search returns results",
        search_ok,
        f"tested {len(SEARCH_TEST_QUERIES)} queries",
    ))

    # --- Check 5: Embedding dimensions ---
    dim_ok = False
    if hasattr(graph_store, "vec_db"):
        try:
            info = graph_store.vec_db.client.get_collection(
                graph_store.vec_db.config.collection_name
            )
            dim = info.config.params.vectors.size
            dim_ok = dim == 1024
            results.append(check(
                "Embedding dimensions",
                dim_ok,
                f"expected=1024, actual={dim}",
            ))
        except Exception as e:
            results.append(check("Embedding dimensions", False, str(e)))
    else:
        results.append(check("Embedding dimensions", False, "no vec_db"))

    # --- Check 6: cube_id on Qdrant points ---
    user_ok = False
    if hasattr(graph_store, "vec_db") and qdrant_count > 0:
        try:
            # Spot-check: get a few points and verify user_name
            items = graph_store.vec_db.get_by_filter(
                {"user_name": cube_id}, scroll_limit=5
            )
            user_ok = len(items) > 0 and all(
                item.payload.get("user_name") == cube_id for item in items
            )
            results.append(check(
                f"cube_id={cube_id} on Qdrant points",
                user_ok,
                f"checked {len(items)} points",
            ))
        except Exception as e:
            results.append(check(f"cube_id={cube_id} on Qdrant points", False, str(e)))
    else:
        results.append(check(
            f"cube_id={cube_id} on Qdrant points",
            False,
            "no points or no vec_db",
        ))

    # --- Check 7: PARENT edges exist ---
    parent_edges = 0
    try:
        for edge in data.get("edges", []):
            if edge.get("type") == "PARENT":
                parent_edges += 1
        results.append(check(
            "PARENT edges exist",
            parent_edges > 0,
            f"found {parent_edges} PARENT edges",
        ))
    except Exception as e:
        results.append(check("PARENT edges exist", False, str(e)))

    # --- Check 8: Scheduler status ---
    scheduler_ok = False
    try:
        req = urllib.request.Request("http://localhost:8000/product/scheduler/allstatus")
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_data = json.loads(resp.read())
            scheduler_ok = isinstance(resp_data, (dict, list))
            detail = json.dumps(resp_data)[:120] if resp_data else "empty response"
            results.append(check("Scheduler status", scheduler_ok, detail))
    except Exception as e:
        results.append(check("Scheduler status", False, str(e)))

    # --- Summary ---
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed}/{total} checks passed")
    print(f"{'=' * 60}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_restore.py <backup_directory>")
        print("Example: python verify_restore.py /app/backups/2025-01-15T1430")
        sys.exit(1)

    run_verification(sys.argv[1])
    print("VERIFY_DONE")  # Marker for shell wrapper to detect completion
