import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Falta variable de entorno: {name}")
    return v


def main():
    sub = env("AZ_SUBSCRIPTION_ID")
    rg = env("AZ_RESOURCE_GROUP")
    ws = env("AZ_WORKSPACE_NAME")
    endpoint_name = os.getenv("AML_ENDPOINT_NAME", "cost-forecast-ets")

    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    ml_client = MLClient(credential, sub, rg, ws)

    ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
    print("Deleted endpoint:", endpoint_name)


if __name__ == "__main__":
    main()
