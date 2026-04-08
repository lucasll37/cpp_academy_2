"""
bucket_demo.py — Demonstração de consumo do MinIO via boto3.

Lê credenciais do .env na raiz do projeto e realiza operações
básicas nos buckets shared, readonly e na pasta pessoal do usuário.
"""

import io
import json
import os
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Carrega .env da raiz do projeto (um nível acima de python/)
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Configuração ───────────────────────────────────────────────────────────────

ENDPOINT   = os.getenv("MINIO_ENDPOINT", "localhost:9000")
ACCESS_KEY = os.getenv("MINIO_SVC_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SVC_SECRET_KEY")
SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"
USERNAME   = os.getenv("MINIO_SVC_USERNAME")   # usado para o bucket pessoal

if not all([ACCESS_KEY, SECRET_KEY, USERNAME]):
    raise EnvironmentError(
        "Defina MINIO_SVC_ACCESS_KEY, MINIO_SVC_SECRET_KEY e MINIO_SVC_USERNAME no .env"
    )

scheme = "https" if SECURE else "http"

s3 = boto3.client(
    "s3",
    endpoint_url=f"{scheme}://{ENDPOINT}",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def upload_json(bucket: str, key: str, data: dict) -> None:
    payload = json.dumps(data, indent=2).encode()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
    )
    print(f"  ✓ upload  → s3://{bucket}/{key}")


def download_json(bucket: str, key: str) -> dict:
    response = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(response["Body"].read())
    print(f"  ✓ download ← s3://{bucket}/{key}")
    return data


def list_objects(bucket: str, prefix: str = "") -> list[str]:
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [o["Key"] for o in response.get("Contents", [])]


def delete_object(bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)
    print(f"  ✓ delete   → s3://{bucket}/{key}")


# ── Demo ───────────────────────────────────────────────────────────────────────

def demo_shared():
    print("\n── Bucket: shared (leitura e escrita) ──")
    upload_json("shared", "experimento-01/config.json", {
        "modelo": "mlp",
        "epocas": 100,
        "lr": 0.001,
    })
    data = download_json("shared", "experimento-01/config.json")
    print(f"  conteúdo: {data}")
    objetos = list_objects("shared", prefix="experimento-01/")
    print(f"  objetos:  {objetos}")
    delete_object("shared", "experimento-01/config.json")


def demo_readonly():
    print("\n── Bucket: readonly (somente leitura) ──")
    objetos = list_objects("readonly")
    if objetos:
        data = download_json("readonly", objetos[0])
        print(f"  conteúdo: {data}")
    else:
        print("  (vazio — aguardando conteúdo do admin)")

    try:
        upload_json("readonly", "tentativa.json", {"erro": "esperado"})
    except ClientError as e:
        print(f"  ✓ escrita bloqueada como esperado: {e.response['Error']['Code']}")


def demo_personal():
    print(f"\n── Bucket: users/{USERNAME}/ (pasta pessoal) ──")
    key = f"{USERNAME}/resultado-42.json"
    upload_json("users", key, {
        "acuracia": 0.97,
        "loss": 0.032,
        "epocas": 50,
    })
    data = download_json("users", key)
    print(f"  conteúdo: {data}")
    objetos = list_objects("users", prefix=f"{USERNAME}/")
    print(f"  objetos:  {objetos}")
    delete_object("users", key)

    # Tenta acessar pasta de outro usuário
    try:
        list_objects("users", prefix="outro-usuario/")
        print("  ⚠ acesso indevido a pasta alheia!")
    except ClientError as e:
        print(f"  ✓ acesso a pasta alheia bloqueado: {e.response['Error']['Code']}")


if __name__ == "__main__":
    print("=== MinIO Bucket Demo ===")
    demo_shared()
    demo_readonly()
    demo_personal()
    print("\n✓ Demo concluído.")
