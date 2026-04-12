import os
import pandas as pd
from sklearn.model_selection import train_test_split

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATA_URL = "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv"
RAW_DIR = "src/data/raw"
CLEAN_DIR = "src/data/clean"
TRAIN_DIR = "src/data/train"
TEST_DIR = "src/data/test"

RAW_FILE = os.path.join(RAW_DIR, "credit_defaults_raw.csv")
CLEAN_FILE = os.path.join(CLEAN_DIR, "credit_defaults_clean.csv")
TRAIN_FILE = os.path.join(TRAIN_DIR, "credit_defaults_train.csv")
TEST_FILE = os.path.join(TEST_DIR, "credit_defaults_test.csv")

# Azure ML workspace
from src.auth import getMLClient
ml_client = getMLClient(None)

# ---------------------------------------------------------
# Step 1 — Download raw dataset
# ---------------------------------------------------------
os.makedirs(RAW_DIR, exist_ok=True)

if not os.path.exists(RAW_FILE):
    print("Downloading raw dataset...")
    df_raw = pd.read_csv(DATA_URL)
    df_raw.to_csv(RAW_FILE, index=False)
else:
    print("Raw dataset exists — skipping download.")
    df_raw = pd.read_csv(RAW_FILE)

# ---------------------------------------------------------
# Step 2 — Clean dataset (remove junk header row)
# ---------------------------------------------------------
os.makedirs(CLEAN_DIR, exist_ok=True)

if not os.path.exists(CLEAN_FILE):
    print("Cleaning dataset...")
    df = pd.read_csv(RAW_FILE, header=1, index_col=0)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"Saved clean dataset to {CLEAN_FILE}")
else:
    print("Clean dataset exists — skipping cleaning.")
    df = pd.read_csv(CLEAN_FILE)

# ---------------------------------------------------------
# Step 3 — Split into train/test
# ---------------------------------------------------------
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

if not (os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE)):
    print("Creating train/test split...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
else:
    print("Train/test files exist — skipping split.")

# ---------------------------------------------------------
# Step 4 — Create MLTable files
# ---------------------------------------------------------
def write_mltable(path, csv_name):
    mltable_path = os.path.join(path, "MLTable")
    if not os.path.exists(mltable_path):
        with open(mltable_path, "w") as f:
            f.write("paths:\n")
            f.write(f"  - file: ./{csv_name}\n")
            f.write("\n")
            f.write("transformations:\n")
            f.write("  - read_delimited:\n")
            f.write("      delimiter: \",\"\n")
            f.write("      encoding: \"utf8\"\n")
            f.write("      header: true\n")
        print(f"Created MLTable at {mltable_path}")
    else:
        print(f"MLTable already exists at {mltable_path}")

write_mltable(TRAIN_DIR, os.path.basename(TRAIN_FILE))
write_mltable(TEST_DIR, os.path.basename(TEST_FILE))

# ---------------------------------------------------------
# Step 5 — Register datasets
# ---------------------------------------------------------
def register_dataset(name, path):
    print(f"Registering dataset '{name}' from path '{path}'")
    try:
        data_asset = Data(
            name=name,
            path=path,
            type=AssetTypes.MLTABLE,
            #version="v1" # auto increase version
        )
        dataset=ml_client.data.create_or_update(data_asset)
        print(f"Registered dataset: \n\t{dataset.id} \n\tname: {dataset.name} version: {dataset.version}")
    except Exception as e:
        print(f"Failed to register {name}: {e}")
        raise e

register_dataset("credit_defaults_test", TEST_DIR)
register_dataset("credit_defaults_train", TRAIN_DIR)

print("All steps completed.")