import sys
from typing import Union

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from category_encoders.binary import BinaryEncoder

from hotel.constant.training_pipeline import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from hotel.entity.config_entity import DataTransformationConfig
from hotel.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from hotel.exception import HotelException
from hotel.logger import logging
from hotel.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from sklearn.pipeline import Pipeline
from hotel.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise HotelException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HotelException(e, sys)

    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")


            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            bin_columns = self._schema_config['bin_columns']
            num_features = self._schema_config['num_features']
            ordinal_features = self._schema_config["ordinal_columns"]
            custom_features = self._schema_config["custom_features_columns"]

            logging.info("Initialize PowerTransformer")
            # custom_feature_handler = Pipeline(steps=[
            #     ("CustomFeatureHandler", custom_feature_handler)
            # ])

            ordinal_encoder = Pipeline(steps=[
                ("OrdinalEncoder", OrdinalEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            oh_transformer = Pipeline(steps=[
                ("OneHotEncoder", OneHotEncoder())
            ])

            bin_encoder = Pipeline(steps=[
                ("Binary_Encoder", BinaryEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            num_encoder = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                [
                    # ("CustomFeatureHandler",custom_feature_handler , custom_features),
                    ("OrdinalEncoder", ordinal_encoder, ordinal_features),
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Binary_Encoder", bin_encoder, bin_columns),
                    ("Numerical_Encoder", num_encoder, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise HotelException(e, sys) from e

    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            logging.info("Got train features and test features of Training dataset")

            drop_cols = self._schema_config['drop_columns']

            logging.info("drop the columns in drop_cols of Training dataset")

            input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)

            
            target_feature_train_df = target_feature_train_df.replace(
                TargetValueMapping()._asdict()
            )


            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

            target_feature_test_df = test_df[TARGET_COLUMN]


            input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)

            logging.info("drop the columns in drop_cols of Test dataset")

            target_feature_test_df = target_feature_test_df.replace(
              TargetValueMapping()._asdict()
            )

            logging.info("Got train features and test features of Testing dataset")

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

            print(input_feature_train_arr)


            logging.info(
                "Used the preprocessor object to fit transform the train features"
            )

            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Used the preprocessor object to transform the test features")

            logging.info("Created train array and test array")

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saved the preprocessor object")

            logging.info(
                "Exited initiate_data_transformation method of Data_Transformation class"
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise HotelException(e, sys) from e