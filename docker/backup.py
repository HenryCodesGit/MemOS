"""
Backup script for MemOS memory data.

Runs inside the memos container via:
    docker exec memos-api-docker python /app/docker/backup.py [TIMESTAMP]

Exports:
  1. textual_memory.json  — full Neo4j graph (nodes + edges + embeddings)
  2. memories_readable.md — human-readable Markdown summary
  3. backup_manifest.json — counts, timestamp, SHA-256 checksum
"""

import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from memos.api.handlers.component_init import init_server
from memos.log import get_logger

logger = get_logger(__name__)

BACKUP_ROOT = "/app/backups"


def get_cube_id() -> str:
    cube_id = os.environ.get("MOS_CUBE_ID")
    if not cube_id:
        print("ERROR: MOS_CUBE_ID environment variable is not set.")
        print("Set it in your .env file (e.g. MOS_CUBE_ID=my-kb)")
        sys.exit(1)
    return cube_id


def run_backup(timestamp: str) -> None:
    cube_id = get_cube_id()
    backup_dir = os.path.join(BACKUP_ROOT, timestamp)
    os.makedirs(backup_dir, exist_ok=True)

    # --- Step 1: Initialize server components ---
    print("Initializing server components...")
    components = init_server()
    naive = components["naive_mem_cube"]
    text_mem = naive.text_mem
    graph_store = text_mem.graph_store

    # --- Step 2: Export graph with embeddings ---
    print(f"Exporting graph for cube_id={cube_id}...")
    json_data = graph_store.export_graph(
        include_embedding=True,
        user_name=cube_id,
    )

    nodes = json_data.get("nodes", [])
    edges = json_data.get("edges", [])
    print(f"Exported {len(nodes)} nodes, {len(edges)} edges")

    # --- Step 3: Fix embedding format (string -> list) ---
    embedding_count = 0
    for node in nodes:
        metadata = node.get("metadata", {})
        if "embedding" in metadata and isinstance(metadata["embedding"], str):
            with contextlib.suppress(json.JSONDecodeError):
                metadata["embedding"] = json.loads(metadata["embedding"])
                embedding_count += 1
    if embedding_count:
        print(f"Fixed {embedding_count} string-encoded embeddings")

    # --- Step 4: Get Qdrant point count for manifest ---
    qdrant_point_count = 0
    try:
        if hasattr(graph_store, "vec_db"):
            qdrant_point_count = graph_store.vec_db.count(
                {"user_name": cube_id}
            )
            print(f"Qdrant points for {cube_id}: {qdrant_point_count}")
    except Exception as e:
        print(f"Warning: Could not get Qdrant count: {e}")

    # --- Step 5: Write textual_memory.json ---
    memory_file = os.path.join(backup_dir, "textual_memory.json")
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {memory_file}")

    # --- Step 6: Compute SHA-256 ---
    sha256 = hashlib.sha256()
    with open(memory_file, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()

    # --- Step 7: Write human-readable Markdown ---
    md_file = os.path.join(backup_dir, "memories_readable.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Memory Backup — {timestamp}\n\n")
        f.write(f"- Cube ID: {cube_id}\n")
        f.write(f"- Nodes: {len(nodes)}\n")
        f.write(f"- Edges: {len(edges)}\n")
        f.write(f"- Qdrant points: {qdrant_point_count}\n\n")
        f.write("---\n\n")

        for i, node in enumerate(nodes, 1):
            metadata = node.get("metadata", {})
            memory = node.get("memory", "N/A")
            mem_type = metadata.get("memory_type", "unknown")
            key = metadata.get("key", "")
            tags = metadata.get("tags", [])
            status = metadata.get("status", "")
            created = metadata.get("created_at", "")
            updated = metadata.get("updated_at", "")
            node_id = node.get("id", "")

            f.write(f"## [{i}] {mem_type}")
            if key:
                f.write(f" — {key}")
            f.write("\n\n")
            f.write(f"**ID:** `{node_id}`\n\n")
            if tags:
                tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
                f.write(f"**Tags:** {tag_str}\n\n")
            if status:
                f.write(f"**Status:** {status}\n\n")
            f.write(f"**Created:** {created}  \n")
            f.write(f"**Updated:** {updated}\n\n")
            f.write(f"**Content:**\n\n{memory}\n\n")
            has_embedding = (
                "embedding" in metadata
                and metadata["embedding"] is not None
            )
            f.write(f"**Has embedding:** {'Yes' if has_embedding else 'No'}\n\n")
            f.write("---\n\n")

    print(f"Saved: {md_file}")

    # --- Step 8: Write backup manifest ---
    manifest = {
        "timestamp": timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cube_id": cube_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "qdrant_point_count": qdrant_point_count,
        "sha256": checksum,
        "files": [
            "textual_memory.json",
            "memories_readable.md",
            "backup_manifest.json",
        ],
    }
    manifest_file = os.path.join(backup_dir, "backup_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {manifest_file}")

    print(f"\nBackup complete: {backup_dir}")
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}, "
          f"Qdrant: {qdrant_point_count}, SHA-256: {checksum[:16]}...")


if __name__ == "__main__":
    ts = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%dT%H%M")
    run_backup(ts)
    print("BACKUP_DONE")  # Marker for shell wrapper to detect completion
