# pull_feedback.py

import requests
import json
import os
from datetime import datetime

API_URL = "https://dev.macrostrat.org/api/pg/kg_context_entities?version_id=is.null"


def deduplicate_by_hash(data):
    seen = {}
    for item in data:
        key = item.get("hashed_text")
        if key and key not in seen:
            seen[key] = item
    return list(seen.values())


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = "feedback"
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(
        output_folder,
        f"feedback_{timestamp}.json"
    )

    print("Pulling feedback from API...")
    response = requests.get(API_URL)
    response.raise_for_status()

    raw_data = response.json()

    print(f"Pulled {len(raw_data)} records")

    cleaned_data = deduplicate_by_hash(raw_data)

    print(f"{len(cleaned_data)} records after deduplication")

    with open(output_file, "w") as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()