# pull_feedback.py

import requests
import json
import os
import argparse
from datetime import datetime


API_URL = "https://dev.macrostrat.org/api/pg/feedback"


# -------------------------------------------------------------------
# Relation mapping
# -------------------------------------------------------------------

def classify_relation(child):
    child_type = child.get("type")
    if not child_type:
        return "related_to"
    return f"has_{child_type.lower().strip()}"


def map_relations(entities, relations):
    entity_map = {e["id"]: e for e in entities}

    mapped = []

    for r in relations:
        head = entity_map.get(r.get("head"))
        tail = entity_map.get(r.get("tail"))

        if not head or not tail:
            continue

        mapped.append({
            "head": r["head"],
            "relation": classify_relation(tail),
            "tail": r["tail"]
        })

    return mapped


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="feedback")
    parser.add_argument("--total_limit", type=int)
    parser.add_argument("--chunk_size", type=int, default=1000)

    parser.add_argument(
        "--start_id",
        type=int,
        help="Start after this id (keyset pagination)"
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# API fetch
# -------------------------------------------------------------------

def fetch_batch(last_id, chunk_size):
    params = {
        "version_id": "is.null",
        "limit": chunk_size,
        "order": "id.asc"
    }

    if last_id is not None:
        params["id"] = f"gt.{last_id}"

    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    return response.json()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    last_id = args.start_id
    batch_num = 1
    total_saved = 0

    while True:
        print(f"\nFetching batch {batch_num} (last_id={last_id})...")

        batch = fetch_batch(last_id, args.chunk_size)

        if not batch:
            print("No more data. Done.")
            break

        print(f"Pulled {len(batch)} records")

        # apply total limit
        if args.total_limit is not None:
            remaining = args.total_limit - total_saved

            if remaining <= 0:
                print("Reached total_limit. Done.")
                break

            if len(batch) > remaining:
                batch = batch[:remaining]

        # APPLY RELATION MAPPING HERE
        for item in batch:
            entities = item.get("entities", [])
            relations = item.get("relations", [])

            item["relations"] = map_relations(entities, relations)

        # output file
        file_path = os.path.join(
            args.output_dir,
            f"feedback_{timestamp}_part_{batch_num:03d}.jsonl"
        )

        with open(file_path, "w") as f:
            for item in batch:
                f.write(json.dumps(item) + "\n")

        saved_now = len(batch)
        total_saved += saved_now

        print(f"Saved {saved_now} records → {file_path}")
        print(f"Total saved: {total_saved}")

        # update keyset
        last_id = batch[-1]["id"]
        print(f"Next start_id: {last_id}")

        batch_num += 1

        # stop conditions
        if len(batch) < args.chunk_size:
            print("Final batch reached.")
            break

        if args.total_limit is not None and total_saved >= args.total_limit:
            print("Reached total_limit.")
            break

    print("\nDone")


if __name__ == "__main__":
    main()