import argparse
import requests
import os
import csv
from datetime import datetime

BASE_URL = "https://dev.macrostrat.org/api/pg/kg_terms"
BATCH_SIZE = 5000


def fetch_all_terms(output_dir):

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir,
        f"all_terms_{timestamp}.csv"
    )

    last_pk = None
    total_rows = 0

    print(f"Saving to: {output_file}")

    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["term_type", "term"])

        while True:
            params = {
                "limit": BATCH_SIZE,
                "order": "pk.asc"
            }

            if last_pk is not None:
                params["pk"] = f"gt.{last_pk}"

            print(f"Requesting batch after pk={last_pk}")

            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()

            data = response.json()

            if not data:
                print("No more records.")
                break

            for row in data:
                term = row.get("name")
                term_type = row.get("type")

                if term and term_type:
                    writer.writerow([term_type, term])
                    total_rows += 1

            last_pk = data[-1]["pk"]

            print(f"Fetched {len(data)} rows (total written={total_rows})")

            if len(data) < BATCH_SIZE:
                break

    print(f"\nFinished. Total rows written: {total_rows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Folder where CSV should be saved"
    )

    args = parser.parse_args()

    fetch_all_terms(args.output_dir)