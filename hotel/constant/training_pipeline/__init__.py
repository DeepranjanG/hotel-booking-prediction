# pipeline name and root directory constant
import os
from hotel.constant.s3_bucket import TRAINING_BUCKET_NAME
from datetime import date

TARGET_COLUMN = "is_canceled"
PIPELINE_NAME: str = "hotel"
ARTIFACT_DIR: str = "artifact"
CURRENT_YEAR = date.today().year
# common file name

FILE_NAME: str = "hotel.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
ANN_MODEL_FILE_NAME = "model.h5"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
MODEL_FILE_PATH = os.path.join("config", "model.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "hotel"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

"""
ANN MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
# ANN_MODEL_TRAINER_DIR_NAME: str = "ann_model_trainer"
# ANN_MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
# ANN_MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.h5"
# ANN_MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
# ANN_MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")


"""
MODEL Evaluation related constant start with MODEL_EVALUATION var name
"""

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02

MODEL_PUSHER_BUCKET_NAME = TRAINING_BUCKET_NAME
MODEL_PUSHER_S3_KEY = "model-registry"