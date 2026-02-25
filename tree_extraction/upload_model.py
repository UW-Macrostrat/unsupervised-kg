import argparse
import os
from huggingface_hub import HfApi, upload_folder


def main(args):
    token = args.hub_token or os.environ.get("HF_TOKEN")

    if token is None:
        raise ValueError(
            "No Hugging Face token provided. "
            "Use --hub_token or set HF_TOKEN environment variable."
        )

    if not os.path.isdir(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")

    print(f"\nPreparing to upload:")
    print(f"  Local path : {args.model_path}")
    print(f"  Repo ID    : {args.hub_repo}")

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=args.hub_repo,
        token=token,
        exist_ok=True
    )

    # Upload folder
    upload_folder(
        repo_id=args.hub_repo,
        folder_path=args.model_path,
        token=token,
        commit_message=args.commit_message,
    )

    print("\n✅ Upload complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a local model to Hugging Face Hub")

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to local model folder (e.g. models/latest)"
    )

    parser.add_argument(
        "--hub_repo",
        required=True,
        help="HF repo id like username/model-name"
    )

    parser.add_argument(
        "--hub_token",
        default=None,
        help="Hugging Face token (optional if HF_TOKEN env var is set)"
    )

    parser.add_argument(
        "--commit_message",
        default="Upload model",
        help="Commit message for this upload"
    )

    args = parser.parse_args()
    main(args)