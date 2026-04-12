from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

from src.auth import getMLClient
ml_client = getMLClient(None)

reg = False
try:
    registered = ml_client.models.get(name="credit_defaults_model", version="1")
    reg = True
    print("Found already registerd model:", registered.name, registered.version, registered.path)
except Exception as e:
    print("Error: ", e)

if not reg:
    model = Model(
        name="credit_defaults_model",
        version="1",
        path="models/credit_defaults_model",
        type="mlflow_model",
    )

    registered = ml_client.models.create_or_update(model)
    print("Model registered:", registered.name, registered.version, registered.path)


# model = Model(
#     name="credit_defaults_model",
#     version="1",
#     path="models/credit_defaults_model",
#     type="mlflow_model",
#     registry_name="azureml"
# )

# ml_client.models.create_or_update(model)

