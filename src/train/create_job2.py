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
from azure.ai.ml.entities import Environment

from src.auth import getMLClient
import time

# python -m src.train.create_job2

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
    return compute_instance_name

def create_environment(ml_client, custom_env_name = "aml-scikit-learn", env_version = "v1-Initial"):
    try:
        custom_job_env = ml_client.environments.get(custom_env_name, env_version)
        print(f"Environment already exists: {custom_job_env.name}:{custom_job_env.version}")
    except Exception:
        print("Creating environment...")
        custom_job_env = Environment(
            name=custom_env_name,
            description="Custom environment for Credit Card Defaults job",
            tags={"scikit-learn": "1.0.2"},
            conda_file="src/trainenv.yaml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            version=env_version
        )
        custom_job_env = ml_client.environments.create_or_update(custom_job_env)
    print(
        f"Environment with name {custom_job_env.name} is registered to workspace, the environment version is {custom_job_env.version}"
    )
    return custom_env_name, custom_job_env

def create_and_submit_job(ml_client, custom_env_name, compute_instance_name):
    registered_model_name = "credit_defaults_model"
    job = command(
        inputs=dict(
            data=Input(
                type="uri_file",
                path="https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
            ),
            test_train_ratio=0.2,
            learning_rate=0.25,
            registered_model_name=registered_model_name,
        ),
        code="./src/train", # location of source code
        command="python main_train.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --learning_rate ${{inputs.learning_rate}} --registered_model_name ${{inputs.registered_model_name}}",
        environment=f"{custom_env_name}@latest",
        display_name="credit_default_prediction",
        compute=compute_instance_name,
    ) # serverless or define compute
    run = ml_client.jobs.create_or_update(job)
    print("Submitted job:", run.name)

    wait_for_job(ml_client, run.name, 10)
    if job.status in ["Failed", "Canceled"]:
        print(f"Job did not competed succesfully '{job.status}'")
        return
    from azure.ai.ml.entities import Model

    model_uri = f"runs:/{run.id}/model"

    registered_model = Model(
        name=registered_model_name,
        version="1",
        type="mlflow_model",
        path=model_uri,
    )

    ml_client.models.create_or_update(registered_model)
    print(f"Registered model as {registered_model_name}:1")


if __name__ == "__main__":
    ml_client = getMLClient(None)
    print("Creating training environment, if doesn't exist...")
    custom_env_name, custom_job_env = create_environment(ml_client, env_version="1")
    print("Creating computer instance, if doesn't exist...")
    compute_instance_name = create_compute(ml_client)
    create_and_submit_job(ml_client, custom_env_name, compute_instance_name)
