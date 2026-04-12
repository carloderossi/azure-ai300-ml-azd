# src/register_data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from auth import getMLClient

ml_client = getMLClient(None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_raw_dir = os.path.join(BASE_DIR, "data", "adult_raw")
train_dir = os.path.join(BASE_DIR, "data", "adult_train")
test_dir = os.path.join(BASE_DIR, "data", "adult_test")

os.makedirs(data_raw_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

adult_csv_path = os.path.join(data_raw_dir, "adult.csv")

if not os.path.exists(adult_csv_path):
    import urllib.request
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    print(f"Downloading {url} -> {adult_csv_path}")
    urllib.request.urlretrieve(url, adult_csv_path)

print("Loading adult.csv...")
df = pd.read_csv(adult_csv_path)

train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

train_csv_path = os.path.join(train_dir, "adult_train.csv")
test_csv_path = os.path.join(test_dir, "adult_test.csv")

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

mltable_train_path = os.path.join(train_dir, "MLTable")
mltable_test_path = os.path.join(test_dir, "MLTable")

if not os.path.exists(mltable_train_path):
    with open(mltable_train_path, "w") as f:
        f.write(
            "paths:\n"
            "  - file: ./adult_train.csv\n"
            "transformations:\n"
            "  - read_delimited:\n"
            "      delimiter: \",\"\n"
        )

if not os.path.exists(mltable_test_path):
    with open(mltable_test_path, "w") as f:
        f.write(
            "paths:\n"
            "  - file: ./adult_test.csv\n"
            "transformations:\n"
            "  - read_delimited:\n"
            "      delimiter: \",\"\n"
        )

train_data = Data(
    name="adult_train",
    version="1",
    type=AssetTypes.MLTABLE,
    path=train_dir,
)

test_data = Data(
    name="adult_test",
    version="1",
    type=AssetTypes.MLTABLE,
    path=test_dir,
)

print("Registering adult_train...")
ml_client.data.create_or_update(train_data)

print("Registering adult_test...")
ml_client.data.create_or_update(test_data)

print("Done. Registered: adult_train:1 and adult_test:1")