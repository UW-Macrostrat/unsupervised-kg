# pull_feedback.py

import requests
import json
import os
import argparse
from datetime import datetime

API_URL = "https://dev.macrostrat.org/api/pg/kg_context_entities"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="feedback",
        help="Directory to save output files"
    )

    parser.add_argument(
        "--total_limit",
        type=int,
        help="Total number of records to save"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of records per API call"
    )

    parser.add_argument(
        "--start_model_run",
        type=int,
        help="Start after this model_run (keyset pagination)"
    )

    return parser.parse_args()


def fetch_batch(last_model_run, chunk_size):
    params = {
        "version_id": "is.null",
        "limit": chunk_size,
        "order": "model_run.asc"
    }

    if last_model_run is not None:
        params["model_run"] = f"gt.{last_model_run}"

    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    return response.json()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    last_model_run = args.start_model_run
    batch_num = 1
    total_saved = 0

    while True:
        print(f"\nFetching batch {batch_num} (last_model_run={last_model_run})...")

        batch = fetch_batch(last_model_run, args.chunk_size)

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

        # output file (one per batch)
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
        last_model_run = batch[-1]["model_run"]
        print(f"Next start_model_run: {last_model_run}")

        batch_num += 1

        # stop conditions
        if len(batch) < args.chunk_size:
            print("Final batch reached (API returned fewer than requested).")
            break

        if args.total_limit is not None and total_saved >= args.total_limit:
            print("Reached total_limit. Done.")
            break

    print("\nDone")


if __name__ == "__main__":
    main()