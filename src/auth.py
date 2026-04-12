from datetime import datetime

from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    CredentialUnavailableError,
)
from azure.core.exceptions import AzureError
from azure.ai.ml import MLClient

import json
from pathlib import Path

# ANSI colour codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

def dump_user(credentials):
    import requests
    from azure.identity import DefaultAzureCredential
    # Request Graph token
    token = credentials.get_token("https://graph.microsoft.com/.default")

    headers = {
        "Authorization": f"Bearer {token.token}",
        "Content-Type": "application/json"
    }

    # Call Microsoft Graph
    response = requests.get(
        "https://graph.microsoft.com/v1.0/me",
        headers=headers
    )

    user = response.json()
    print(f"{CYAN}User display name:{RESET} {GREEN}{user.get('displayName')}{RESET}")
    print(f"{CYAN}User object ID:{RESET} {YELLOW}{user.get('id')}{RESET}")
    print(f"{CYAN}User email:{RESET} {MAGENTA}{user.get('mail') or user.get('userPrincipalName')}{RESET}")

def load_config(path: str) -> dict:
    """
    Load environment variables based on the default environment defined in .azure/config.json.

    Steps:
    - Resolve the caller's directory using __file__
    - Locate ../../.azure relative to that file
    - Read defaultEnvironment from .azure/config.json
    - Load variables from .azure/{defaultEnvironment}/.env
    - Return them as a dict
    """
    # Resolve the directory of the file where this function lives
    base_dir = Path(__file__).resolve().parents[1]
    print(f"{CYAN}Project Base_DIR:{RESET} {GREEN}{base_dir}{RESET}")

    # Locate the .azure folder relative to the file
    azure_dir = (base_dir / ".azure").resolve()
    print(f"{CYAN}'.azure' folder:{RESET} {GREEN}{azure_dir}{RESET}")

    config_path = azure_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at: {config_path}")

    # Load config.json
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    default_env = config.get("defaultEnvironment")
    if not default_env:
        raise ValueError("defaultEnvironment not found in config.json")

    # Path to the .env file
    env_path = azure_dir / default_env / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file at: {env_path}")

    # Parse .env file
    env_vars = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            env_vars[key.strip()] = value

    return env_vars

def get_credentials(config_path) -> DefaultAzureCredential:
    """
    Returns a valid Azure credential for the current environment.

    Resolution order:
      1. DefaultAzureCredential  — covers env vars, workload identity,
                                    managed identity, Azure CLI, VS Code,
                                    Azure PowerShell, azd CLI.
      2. InteractiveBrowserCredential — fallback for local dev when none
                                         of the above are configured.
    """
    config = load_config(config_path)
    tenant_id = config["AZURE_TENANT_ID"]
    print(f"{CYAN}Using tenant_id:{RESET} {YELLOW}{tenant_id}")
    print(f"{RESET}")

    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        token = credential.get_token("https://management.azure.com/.default")
        print("Successfully obtained credentials using DefaultAzureCredential.")
        expiry = datetime.fromtimestamp(token.expires_on)
        print("Access token expires at:", expiry)
    except Exception as ex:
        print("Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work")
        #credential = InteractiveBrowserCredential()
    try:
        if not credential:
            credential = InteractiveBrowserCredential(
                tenant_id=tenant_id, process_timeout=10,
            )

            """         credential = DefaultAzureCredential(
                # Silence the individual "unavailable" messages in the chain
                logging_enable=False,
                # Give each probe a short timeout so the fallback is snappy
                process_timeout=5,
            ) """
            # Eagerly probe so we fail-fast here rather than mid-operation
        credential.get_token("https://management.azure.com/.default")
        try:
            dump_user(credential)
        except Exception as e:
            print("An error occurred dumping the user: ", e)        
        return credential
    except (AzureError, CredentialUnavailableError):
        print("InteractiveBrowserCredential failed to obtain a token.")
        raise RuntimeError("Failed to obtain Azure credentials using both DefaultAzureCredential and InteractiveBrowserCredential. Please check your configuration and environment.")

def getMLClient(config_path: str):
    get_credentials(None)
    config = load_config(config_path)
    SUBSCRIPTION_ID = config["AZURE_SUBSCRIPTION_ID"]
    print(f"{CYAN}Using subscription:{RESET} {YELLOW}{SUBSCRIPTION_ID}{RESET}")
    WORKSPACE_NAME = config["AZURE_ML_WORKSPACE"]
    print(f"{CYAN}Using workspace:{RESET} {GREEN}{WORKSPACE_NAME}{RESET}")
    RESOURCE_GROUP = config["AZURE_RESOURCE_GROUP"]
    print(f"{CYAN}Using resource group:{RESET} {YELLOW}{RESOURCE_GROUP}{RESET}")

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    return ml_client

if __name__ == "__main__":
    mlclient = getMLClient(None)
    print(f"{CYAN}Successfully obtained credentials:{RESET} {BLUE}", mlclient)
