"""
One-time bootstrap script.
Reads aws_infra/config/app_configs.json and pushes every entry
into AWS SSM Parameter Store under /model-router/{app_id}/{env}.

Usage:
    python ssm_bootstrap.py [--region us-east-1] [--prefix /model-router]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import boto3

DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_PREFIX = "/model-router"
CONFIG_FILE    = Path(__file__).parent / "app_configs.json"


def push_configs(region: str, prefix: str, dry_run: bool = False) -> None:
    ssm = boto3.client("ssm", region_name=region)

    with open(CONFIG_FILE, "r") as f:
        data = json.load(f)

    apps = data.get("apps", [])
    print(f"Pushing {len(apps)} app config(s) to SSM ({region}) under '{prefix}'...")

    for cfg in apps:
        app_id = cfg["app_id"]
        env    = cfg.get("env", "prod")
        name   = f"{prefix}/{app_id}/{env}"
        value  = json.dumps(cfg)

        if dry_run:
            print(f"  [DRY-RUN] Would write: {name}")
            continue

        try:
            ssm.put_parameter(
                Name=name,
                Value=value,
                Type="String",
                Overwrite=True,
            )
            print(f"  [OK] {name}")
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap SSM with app configs.")
    parser.add_argument("--region",    default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--prefix",    default=DEFAULT_PREFIX, help="SSM path prefix")
    parser.add_argument("--dry-run",   action="store_true",    help="Print without writing")
    args = parser.parse_args()

    push_configs(args.region, args.prefix, dry_run=args.dry_run)
