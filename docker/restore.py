"""
Restore script for MemOS memory data.

Runs inside the memos container via:
    docker exec memos-api-docker python /app/docker/restore.py /app/backups/<timestamp>

Restores from textual_memory.json:
  1. Neo4j nodes + edges via import_graph() (MERGE — idempotent)
  2. Qdrant vectors re-synced from node embeddings (fills the gap import_graph doesn't cover)

Verifies counts against backup_manifest.json when available.
"""

import json
import os
import sys

from memos.api.handlers.component_init import init_server
from memos.log import get_logger
from memos.vec_dbs.item import VecDBItem

logger = get_logger(__name__)


def get_cube_id() -> str:
    cube_id = os.environ.get("MOS_CUBE_ID")
    if not cube_id:
        print("ERROR: MOS_CUBE_ID environment variable is not set.")
        print("Set it in your .env file (e.g. MOS_CUBE_ID=my-kb)")
        sys.exit(1)
    return cube_id


def restore_from_json(backup_dir: str) -> None:
    cube_id = get_cube_id()

    memory_file = os.path.join(backup_dir, "textual_memory.json")
    manifest_file = os.path.join(backup_dir, "backup_manifest.json")

    if not os.path.exists(memory_file):
        print(f"ERROR: {memory_file} not found")
        sys.exit(1)

    # Load backup data
    print(f"Loading backup from {memory_file}...")
    with open(memory_file, encoding="utf-8") as f:
        json_data = json.load(f)

    nodes = json_data.get("nodes", [])
    edges = json_data.get("edges", [])
    print(f"Loaded {len(nodes)} nodes, {len(edges)} edges")

    # Load manifest for verification (optional)
    manifest = None
    if os.path.exists(manifest_file):
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)

    # --- Step 1: Initialize server ---
    print("Initializing server components...")
    components = init_server()
    naive = components["naive_mem_cube"]
    text_mem = naive.text_mem
    graph_store = text_mem.graph_store

    # --- Step 2: Restore Neo4j via import_graph ---
    print(f"Restoring Neo4j graph (cube_id={cube_id})...")
    graph_store.import_graph(json_data, user_name=cube_id)
    print(f"Neo4j restore complete: {len(nodes)} nodes, {len(edges)} edges")

    # --- Step 3: Re-sync Qdrant ---
    # import_graph only restores Neo4j — Qdrant must be re-populated separately.
    if not hasattr(graph_store, "vec_db"):
        print("WARNING: graph_store has no vec_db — skipping Qdrant sync")
    else:
        print("Re-syncing Qdrant vectors...")
        vec_db = graph_store.vec_db

        # Ensure collection exists (it should after server init, but just in case)
        if not vec_db.collection_exists(vec_db.config.collection_name):
            print(f"Creating Qdrant collection: {vec_db.config.collection_name}")
            vec_db.create_collection()

        # Build VecDBItems from backup nodes
        vec_items = []
        skipped = 0
        for node in nodes:
            node_id = node.get("id")
            memory = node.get("memory", "")
            metadata = node.get("metadata", {})
            embedding = metadata.get("embedding")

            if not embedding or not isinstance(embedding, list):
                skipped += 1
                continue

            # Build payload matching neo4j_community.py add_node format
            payload = {"memory": memory, "vector_sync": "success"}
            # Copy all metadata keys except embedding to payload
            for k, v in metadata.items():
                if k == "embedding":
                    continue
                payload[k] = v
            # Ensure cube_id is set
            payload["user_name"] = cube_id

            vec_items.append(VecDBItem(
                id=node_id,
                vector=embedding,
                payload=payload,
            ))

        if vec_items:
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vec_items), batch_size):
                batch = vec_items[i:i + batch_size]
                vec_db.upsert(batch)
                print(f"  Upserted Qdrant batch {i // batch_size + 1} "
                      f"({len(batch)} points)")

        print(f"Qdrant sync complete: {len(vec_items)} points upserted, "
              f"{skipped} skipped (no embedding)")

    # --- Step 4: Verification summary ---
    print("\n--- Verification ---")
    if manifest:
        expected_nodes = manifest.get("node_count", "?")
        expected_edges = manifest.get("edge_count", "?")
        expected_qdrant = manifest.get("qdrant_point_count", "?")
        print(f"Expected: nodes={expected_nodes}, edges={expected_edges}, "
              f"qdrant={expected_qdrant}")

    # Check actual counts
    try:
        actual_data = graph_store.export_graph(
            include_embedding=False,
            user_name=cube_id,
        )
        actual_nodes = actual_data.get("total_nodes", len(actual_data.get("nodes", [])))
        actual_edges = actual_data.get("total_edges", len(actual_data.get("edges", [])))
        print(f"Actual Neo4j: nodes={actual_nodes}, edges={actual_edges}")
    except Exception as e:
        print(f"Warning: Could not verify Neo4j counts: {e}")

    if hasattr(graph_store, "vec_db"):
        try:
            actual_qdrant = graph_store.vec_db.count({"user_name": cube_id})
            print(f"Actual Qdrant: points={actual_qdrant}")
        except Exception as e:
            print(f"Warning: Could not verify Qdrant count: {e}")

    print("\nRestore complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python restore.py <backup_directory>")
        print("Example: python restore.py /app/backups/2025-01-15T1430")
        sys.exit(1)

    restore_from_json(sys.argv[1])
    print("RESTORE_DONE")  # Marker for shell wrapper to detect completion
