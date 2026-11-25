"""
deploy/hf/deploy_to_hf.py
Upload the project + ChromaDB to a Hugging Face Docker Space in one command.

Usage:
    pip install huggingface_hub
    python deploy/hf/deploy_to_hf.py --repo YOUR_USERNAME/financial-advisor
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Deploy AI Financial Advisor to Hugging Face Spaces")
    parser.add_argument("--repo", required=True, help="Space repo ID, e.g. username/financial-advisor")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF write token (or run `huggingface-cli login`)")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    print(f"Ensuring Space '{args.repo}' exists (Docker SDK)...")
    api.create_repo(
        repo_id=args.repo,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    print("Uploading project files + chroma_db (large files handled automatically)...")
    api.upload_folder(
        folder_path=str(PROJECT_ROOT),
        repo_id=args.repo,
        repo_type="space",
        ignore_patterns=[
            ".git/*",
            ".env",
            ".venv/*",
            "venv/*",
            "__pycache__/*",
            "**/__pycache__/*",
            "*.pyc",
            "logs/*",
            "data/*",
            "tests/*",
            ".pytest_cache/*",
            "*.sqlite",
            "*.sqlite3",
            "Dockerfile",
            ".dockerignore",
            "README.md",
            "docker-compose.yml",
            "deploy/k8s/*",
        ],
    )

    # Upload HF-specific root files (Dockerfile, README.md, .dockerignore)
    for filename in ["Dockerfile", "README.md", ".dockerignore"]:
        src = PROJECT_ROOT / "deploy" / "hf" / filename
        print(f"Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=filename,
            repo_id=args.repo,
            repo_type="space",
        )

    print(f"\nDone! View your Space at: https://huggingface.co/spaces/{args.repo}")
    print("Next: Open Settings -> Variables and secrets in your Space and add your .env secrets!")


if __name__ == "__main__":
    main()
