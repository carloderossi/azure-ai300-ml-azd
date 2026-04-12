from azure.ai.ml import MLClient
from auth import getMLClient

ml_client = getMLClient(None)

ml_client.data._data_operations.delete(
    name="adult-raw",
    version="1"
)