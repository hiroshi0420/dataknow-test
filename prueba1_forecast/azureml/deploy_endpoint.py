import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
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
    location = os.getenv("AZ_LOCATION", "eastus")

    endpoint_name = os.getenv("AML_ENDPOINT_NAME", "cost-forecast-ets")
    deployment_name = os.getenv("AML_DEPLOYMENT_NAME", "blue")
    instance_type = os.getenv("AML_INSTANCE_TYPE", "Standard_DS2_v2")
    instance_count = int(os.getenv("AML_INSTANCE_COUNT", "1"))

    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    ml_client = MLClient(credential, sub, rg, ws)

    bundle_path = Path("outputs") / "models" / "ets_materials_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No existe {bundle_path}. Ejecuta primero: python main.py --horizon 36"
        )

    model = Model(
        path=str(bundle_path),
        name=f"{endpoint_name}-bundle",
        description="ETS fitted bundle (X/Y/Z) for cost forecasting",
        type="custom_model",
    )
    model = ml_client.models.create_or_update(model)

    environment = Environment(
        name=f"{endpoint_name}-env",
        conda_file="azureml/conda.yml",
        image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py310-cpu-inference:latest",
    )
    environment = ml_client.environments.create_or_update(environment)

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Cost forecast scoring (ETS)",
        auth_mode="key",
        location=location,
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(code="azureml", scoring_script="score.py"),
        instance_type=instance_type,
        instance_count=instance_count,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    keys = ml_client.online_endpoints.get_keys(endpoint_name)
    endpoint = ml_client.online_endpoints.get(endpoint_name)

    print("\nDEPLOY OK")
    print("Endpoint:", endpoint_name)
    print("Scoring URI:", endpoint.scoring_uri)
    print("Key:", keys.primary_key)


if __name__ == "__main__":
    main()
