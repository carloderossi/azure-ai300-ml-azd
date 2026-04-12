import os
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Environment
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, ComputeInstance
from azure.core.exceptions import ResourceNotFoundError

from src.auth import getMLClient
import time

# python -m src.train.train_job

def wait_for_job(ml_client, job_name, poll_interval=5):
    """Block until the given job enters a terminal state.

    Parameters
    ----------
    job_name : str
        Name of the Azure ML job to poll.
    poll_interval : int, optional
        Seconds to wait between status checks.

    Returns
    -------
    azure.ai.ml.entities.Job
        Final job object once status is Completed, Failed, or Canceled.
    """
    while True:
        job = ml_client.jobs.get(job_name)
        print(f"Job status: {job.status}")
        if job.status in ["Completed", "Failed", "Canceled"]:
            return job
        time.sleep(poll_interval)

def create_dataset(ml_client):
    print("\n=== CREATING DATASET ===")
    dataset_name = "adultraw"
    # datastore = ml_client.datastores.get("workspaceblobstore")

    data_version = "1"
    try:
        data_asset = ml_client.data.get(name=dataset_name, version=data_version)
        print(f"Dataset '{dataset_name}' version '{data_version}' already exists: {data_asset.id}")  
    except Exception as e:
        print(f"Dataset '{dataset_name}' version '{data_version}' not found. Creating new one...")
        # ml_client.data.get(name=dataset_name, version=data_version)
        data_asset = Data(
            name=dataset_name,
            version="1",
            type=AssetTypes.URI_FILE,
            path="./src/data/adult_raw/adult.csv"  # local path
        )

        ml_client.data.create_or_update(data_asset)
        print(f"Dataset '{dataset_name}' created : {data_asset}")

    print(f"Dataset '{dataset_name}' created with id: {data_asset.id}")
    return dataset_name

def create_compute(ml_client, compute_instance_name="ml-ai300-cpu"):
    vm_size = "STANDARD_DS11_V2"
    # Compute Instance constructor only accept name and size
    # region = cfg["region"]
    # tier = cfg.get("compute_tier", "low_priority")

    print("\n=== Creating Compute Instance ===")
    try:
        print(f"Checking if compute instance '{compute_instance_name}' exists...")
        ci = ml_client.compute.get(compute_instance_name) #, workspace_name=workspace.name, resource_group_name=workspace.resource_group)
        print(f"Compute instance '{compute_instance_name}' already exists.")
    except ResourceNotFoundError:
        print(f"Compute instance '{compute_instance_name}' not found. Creating new one...")
        ci = ComputeInstance(
            name=compute_instance_name,
            size=vm_size,
            # location=region,
            # tier=tier, not existing for compute instance
            # workspace_name=workspace.name,
            # resource_group_name=workspace.resource_group
        )
        poller = ml_client.compute.begin_create_or_update(ci)
        ci = poller.result()
        print(f"Compute instance '{compute_instance_name}' created.")

    print("\n=== Compute Resources Ready ===")
    print(f"Compute Instance: {ci}")

ml_client = getMLClient(None)

from azure.ai.ml.entities import Environment

env_name = "logisticRegMLFlow-env"
env_version = "1"

try:
    env = ml_client.environments.get(env_name, env_version)
    print(f"Environment already exists: {env.name}:{env.version}")
except Exception:
    print("Creating environment...")
    env = Environment(
        name=env_name,
        version=env_version,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="src/conda.yml",
    )
    ml_client.environments.create_or_update(env)
    print("Environment created.")

create_compute(ml_client)

dataset_name = create_dataset(ml_client)

print(f"Creating job inputs...")
csv_job_inputs = {
    "input_data": Input(
        type=AssetTypes.URI_FILE,
        path=f"azureml:{dataset_name}:1"
    ),
    "input_format": "csv"   # or "mltable"
}
print(f"Job inputs: {csv_job_inputs}")

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

job = command(
    code="./src/train",
    command="python train.py --inputs ${{inputs.input_data}} --format ${{inputs.input_format}}",
    environment=f"azureml:{env_name}:1",
    experiment_name="adult-train-exp",
    compute="ml-ai300-cpu",
    inputs=csv_job_inputs
)

run = ml_client.jobs.create_or_update(job)
print("Submitted job:", run.name)

wait_for_job(ml_client, run.name, 10)
if job.status in ["Failed", "Canceled"]:
    print(f"Job did not competed succesfully '{job.status}'")
    exit()

from azure.ai.ml.entities import Model

model_uri = f"runs:/{run.id}/model"

registered_model = Model(
    name="adult_model",
    version="1",
    type="mlflow_model",
    path=model_uri,
)

ml_client.models.create_or_update(registered_model)
print("Registered model as adult_model:1")