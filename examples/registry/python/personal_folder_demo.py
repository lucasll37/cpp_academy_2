"""
personal_folder_demo.py — Cria 10 arquivos na pasta pessoal do usuário e lista-os.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import boto3
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Configuração ───────────────────────────────────────────────────────────────

ENDPOINT   = os.getenv("MINIO_ENDPOINT", "localhost:9000")
ACCESS_KEY = os.getenv("MINIO_SVC_ACCESS_KEY")
SECRET_KEY = os.getenv("MINIO_SVC_SECRET_KEY")
USERNAME   = os.getenv("MINIO_SVC_USERNAME")
SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"

if not all([ACCESS_KEY, SECRET_KEY, USERNAME]):
    raise EnvironmentError(
        "Defina MINIO_SVC_ACCESS_KEY, MINIO_SVC_SECRET_KEY e MINIO_SVC_USERNAME no .env"
    )

s3 = boto3.client(
    "s3",
    endpoint_url=f"{'https' if SECURE else 'http'}://{ENDPOINT}",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

BUCKET = "users"
PREFIX = f"{USERNAME}/"

# ── Upload ─────────────────────────────────────────────────────────────────────

print(f"Criando 10 arquivos em s3://{BUCKET}/{PREFIX}\n")

for i in range(1, 11):
    key  = f"{PREFIX}experimento-{i:02d}.json"
    data = {
        "id":       i,
        "acuracia": round(random.uniform(0.80, 0.99), 4),
        "loss":     round(random.uniform(0.01, 0.20), 4),
        "epocas":   random.randint(10, 100),
        "timestamp": datetime.utcnow().isoformat(),
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"  ✓ {key}")

# ── Listagem ───────────────────────────────────────────────────────────────────

print(f"\nObjetos em s3://{BUCKET}/{PREFIX}\n")

response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
objetos  = response.get("Contents", [])

print(f"  {'Chave':<45} {'Tamanho':>10}  {'Modificado'}")
print(f"  {'-'*45}  {'-'*10}  {'-'*20}")

for obj in objetos:
    print(f"  {obj['Key']:<45} {obj['Size']:>9}B  {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n  Total: {len(objetos)} objetos")
