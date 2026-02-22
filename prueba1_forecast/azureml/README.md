# Azure ML Online Endpoint (opcional)

Objetivo: desplegar **solo** el scoring del forecast (ETS) como Online Endpoint.

## Requisitos
- `az login`
- Workspace de Azure ML existente
- Dependencias:

```bash
pip install -r requirements-azureml.txt
```

## Flujo recomendado (minimiza costo)
1) Entrena y genera el bundle localmente (y valida outputs):
```bash
python main.py --horizon 36
```
Esto crea `outputs/models/ets_materials_bundle.joblib`.

2) Despliega endpoint:
```bash
python azureml/deploy_endpoint.py
```

3) Invoca endpoint:
```bash
python azureml/invoke_endpoint.py --horizon 36
```

4) Borra endpoint (corta costo):
```bash
python azureml/delete_endpoint.py
```

## Variables de entorno
Puedes exportarlas o usar `.env` local.

- `AZ_SUBSCRIPTION_ID`
- `AZ_RESOURCE_GROUP`
- `AZ_WORKSPACE_NAME`
- `AZ_LOCATION` (default eastus)

Opcionales:
- `AML_ENDPOINT_NAME` (default: cost-forecast-ets)
- `AML_DEPLOYMENT_NAME` (default: blue)
- `AML_INSTANCE_TYPE` (default: Standard_DS2_v2)
- `AML_INSTANCE_COUNT` (default: 1)
