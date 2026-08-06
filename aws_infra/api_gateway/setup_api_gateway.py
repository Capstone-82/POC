"""
Provisions the Model Router API Gateway infrastructure using boto3.

Creates:
  - A REST API named 'ModelRouterAPI'
  - A POST /invoke resource wired to the given Lambda function
  - A Usage Plan + API Key per application (from app_configs.json)
  - Associates the Usage Plan with the deployed stage

Usage:
    python setup_api_gateway.py \
        --lambda-arn arn:aws:lambda:us-east-1:123456789:function:model-router \
        [--region us-east-1] [--stage prod]
"""

import argparse
import json
import os
import time
from pathlib import Path

import boto3

DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
CONFIG_FILE    = Path(__file__).parent.parent / "config" / "app_configs.json"
API_NAME       = "ModelRouterAPI"


def _wait(secs: float = 1.0):
    time.sleep(secs)


def setup(lambda_arn: str, region: str, stage: str) -> None:
    apigw  = boto3.client("apigateway", region_name=region)
    lam    = boto3.client("lambda",     region_name=region)
    sts    = boto3.client("sts",        region_name=region)
    acc_id = sts.get_caller_identity()["Account"]

    # ── 1. Create REST API ────────────────────────────────────
    print(f"Creating REST API '{API_NAME}'...")
    api   = apigw.create_rest_api(name=API_NAME, description="Model Router Gateway")
    api_id = api["id"]
    print(f"  API ID: {api_id}")

    # ── 2. Get root resource ──────────────────────────────────
    resources = apigw.get_resources(restApiId=api_id)["items"]
    root_id   = next(r["id"] for r in resources if r["path"] == "/")

    # ── 3. Create /invoke resource ────────────────────────────
    print("Creating /invoke resource...")
    invoke_resource = apigw.create_resource(
        restApiId=api_id, parentId=root_id, pathPart="invoke"
    )
    invoke_id = invoke_resource["id"]

    # ── 4. Create POST method (API Key required) ──────────────
    print("Adding POST method (API key required)...")
    apigw.put_method(
        restApiId=api_id,
        resourceId=invoke_id,
        httpMethod="POST",
        authorizationType="NONE",
        apiKeyRequired=True,
    )

    # ── 5. Wire Lambda integration ────────────────────────────
    lambda_uri = (
        f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/"
        f"{lambda_arn}/invocations"
    )
    print(f"Integrating with Lambda: {lambda_arn}")
    apigw.put_integration(
        restApiId=api_id,
        resourceId=invoke_id,
        httpMethod="POST",
        type="AWS_PROXY",
        integrationHttpMethod="POST",
        uri=lambda_uri,
    )

    # ── 6. Grant API Gateway permission to invoke Lambda ──────
    lam.add_permission(
        FunctionName=lambda_arn,
        StatementId=f"apigw-invoke-{api_id}",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=f"arn:aws:execute-api:{region}:{acc_id}:{api_id}/*/POST/invoke",
    )

    # ── 7. Deploy to stage ────────────────────────────────────
    print(f"Deploying to stage '{stage}'...")
    deployment = apigw.create_deployment(restApiId=api_id, stageName=stage)
    _wait()

    # ── 8. Create Usage Plans + API Keys per app ──────────────
    with open(CONFIG_FILE, "r") as f:
        configs = json.load(f)["apps"]

    seen_apps = set()
    for cfg in configs:
        app_id  = cfg["app_id"]
        env     = cfg.get("env", "prod")
        label   = f"{app_id}-{env}"
        if label in seen_apps:
            continue
        seen_apps.add(label)

        throttle = cfg.get("throttle", {})
        rate     = throttle.get("rate_limit",   100)
        burst    = throttle.get("burst_limit",  200)
        quota    = throttle.get("quota_per_day",10000)

        print(f"Creating Usage Plan for '{label}'...")
        plan = apigw.create_usage_plan(
            name=f"UsagePlan-{label}",
            throttle={"rateLimit": float(rate), "burstLimit": burst},
            quota={"limit": quota, "period": "DAY"},
            apiStages=[{"apiId": api_id, "stage": stage}],
        )
        plan_id = plan["id"]

        print(f"Creating API Key for '{label}'...")
        key = apigw.create_api_key(name=f"APIKey-{label}", enabled=True)
        key_id  = key["id"]
        key_val = key["value"]

        apigw.create_usage_plan_key(usagePlanId=plan_id, keyId=key_id, keyType="API_KEY")
        print(f"  API Key value: {key_val}")
        _wait(0.5)

    invoke_url = (
        f"https://{api_id}.execute-api.{region}.amazonaws.com/{stage}/invoke"
    )
    print(f"\nSetup complete.")
    print(f"Invoke URL: {invoke_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision Model Router API Gateway.")
    parser.add_argument("--lambda-arn", required=True,          help="Lambda function ARN")
    parser.add_argument("--region",     default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--stage",      default="prod",          help="Deployment stage name")
    args = parser.parse_args()
    setup(args.lambda_arn, args.region, args.stage)
