from azure.ai.ml import MLClient, Input, dsl
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# from azure.ai.ml.entities import (
#     RAIInsightsConstructor,
#     RAIInsightsGather,
#     RAIInsightsScorecard,
#     RAIInsightsDashboard,
# )

# -------------------------------------------------------------------
# MLClient initialization
# -------------------------------------------------------------------
from src.auth import getMLClient

ml_client = getMLClient(None)

# -------------------------------------------------------------------
# Model + data references
# -------------------------------------------------------------------
model_path = "azureml:credit_defaults_model:1"
train_data = "azureml:credit_defaults_train:4"  # use version 1 in case
test_data = "azureml:credit_defaults_test:4"
target_column = "income"

from azure.ai.ml import load_component

# -------------------------------------------------------------------
# Load RAI Components from the Registry
# -------------------------------------------------------------------
# These names are standard across Azure ML regions
version = "0.0.1" # Or the latest version available in your region
rai_constructor_comp = ml_client.components.get(name="microsoft_azureml_rai_tabular_insight_constructor", version=version)
rai_gather_comp = ml_client.components.get(name="microsoft_azureml_rai_tabular_insight_gather", version=version)
rai_scorecard_comp = ml_client.components.get(name="microsoft_azureml_rai_tabular_insight_scorecard", version=version)

# -------------------------------------------------------------------
# Pipeline definition
# -------------------------------------------------------------------
@dsl.pipeline(compute="cpu-cluster")
def rai_pipeline():

    # 1. Constructor
    constructor_job = rai_constructor_comp(
        title="Credit Defaults RAI Dashboard",
        task_type="classification",
        model_info=model_path,
        model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=model_path),
        train_dataset=Input(type=AssetTypes.MLTABLE, path=train_data),
        test_dataset=Input(type=AssetTypes.MLTABLE, path=test_data),
        target_column_name="income",
        categorical_column_names='[]' # Add if needed
    )

    # 2. Insights
    error_analysis_job = RAIInsightsDashboard(
        constructor=constructor_job.outputs.rai_insights_dashboard,
        insight_type="error_analysis",
    )

    counterfactual_job = RAIInsightsDashboard(
        constructor=constructor_job.outputs.rai_insights_dashboard,
        insight_type="counterfactual",
    )

    causal_job = RAIInsightsDashboard(
        constructor=constructor_job.outputs.rai_insights_dashboard,
        insight_type="causal",
    )

    explanation_job = RAIInsightsDashboard(
        constructor=constructor_job.outputs.rai_insights_dashboard,
        insight_type="explanation",
    )

    # 3. Gather
    gather_job = RAIInsightsGather(
        constructor=constructor_job.outputs.rai_insights_dashboard,
        insights=[
            error_analysis_job.outputs.insight,
            counterfactual_job.outputs.insight,
            causal_job.outputs.insight,
            explanation_job.outputs.insight,
        ],
    )

    # 4. Scorecard
    scorecard_job = RAIInsightsScorecard(
        dashboard=gather_job.outputs.dashboard,
        pdf_generation_config=Input(
            type=AssetTypes.URI_FILE, path="src/rai/pdf_gen.json"
        ),
        predefined_cohorts_json=Input(
            type=AssetTypes.URI_FILE, path="src/rai/cohorts.json"
        ),
    )

    return {
        "dashboard": gather_job.outputs.dashboard,
        "scorecard": scorecard_job.outputs.scorecard,
    }


# -------------------------------------------------------------------
# Submit pipeline
# -------------------------------------------------------------------
if __name__ == "__main__":
    job = rai_pipeline()
    created = ml_client.jobs.create_or_update(job)
    print("Submitted RAI pipeline job:", created.name)