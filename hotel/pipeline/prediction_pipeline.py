import logging
import os
import sys

import numpy as np
import pandas as pd
from hotel.cloud_storage.aws_storage import SimpleStorageService
from hotel.constant.training_pipeline import SCHEMA_FILE_PATH
from hotel.entity.config_entity import HotelPredictorConfig
from hotel.entity.s3_estimator import HotelEstimator
from hotel.exception import HotelException
from hotel.logger import logging
from hotel.utils.main_utils import read_yaml_file
from pandas import DataFrame


class HotelData:
    def __init__(self,
                 hotel,
                 lead_time,
                 arrival_date_year,
                 arrival_date_month,
                 arrival_date_day_of_month,
                 stays_in_weekend_nights,
                 stays_in_week_nights,
                 meal,
                 country,
                 market_segment,
                 distribution_channel,
                 previous_cancellations,
                 reserved_room_type,
                 booking_changes,
                 deposit_type,
                 customer_type,
                 adr,
                 total_of_special_requests,
                 total_guest
                ):
        """
        Hotel Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.hotel = hotel
            self.lead_time = lead_time
            self.arrival_date_year = arrival_date_year
            self.arrival_date_month = arrival_date_month
            self.arrival_date_day_of_month = arrival_date_day_of_month
            self.stays_in_weekend_nights = stays_in_weekend_nights
            self.stays_in_week_nights = stays_in_week_nights
            self.meal = meal
            self.country = country
            self.market_segment = market_segment
            self.distribution_channel = distribution_channel
            self.previous_cancellations = previous_cancellations
            self.reserved_room_type = reserved_room_type
            self.booking_changes = booking_changes
            self.deposit_type = deposit_type
            self.customer_type = customer_type
            self.adr= adr
            self.total_of_special_requests = total_of_special_requests
            self.total_guest = total_guest


        except Exception as e:
            raise HotelException(e, sys) from e

    def get_hotel_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from HotelData class input
        """
        try:
            
            hotel_input_dict = self.get_hotel_data_as_dict()
            return DataFrame(hotel_input_dict)
        
        except Exception as e:
            raise HotelException(e, sys) from e


    def get_hotel_data_as_dict(self):
        """
        This function returns a dictionary from HotelData class input 
        """
        logging.info("Entered get_hotel_data_as_dict method as HotelData class")

        try:
            input_data =  {
                "hotel": [self.hotel],
                "lead_time":[self.lead_time],
                "arrival_date_year":[self.arrival_date_year],
                "arrival_date_month":[self.arrival_date_month],
                "arrival_date_day_of_month":[self.arrival_date_day_of_month],
                "stays_in_weekend_nights":[self.stays_in_weekend_nights],
                "stays_in_week_nights":[self.stays_in_week_nights],
                "meal":[self.meal],
                "country":[self.country],
                "market_segment":[self.market_segment],
                "distribution_channel":[self.distribution_channel],
                "previous_cancellations":[self.previous_cancellations],
                "reserved_room_type":[self.reserved_room_type],
                "booking_changes":[self.booking_changes],
                "deposit_type":[self.deposit_type],
                "customer_type":[self.customer_type],
                "adr":[self.adr],
                "total_of_special_requests":[self.total_of_special_requests],
                "total_guest":[self.total_guest]
                }

            logging.info("Created hotel data dict")

            logging.info("Exited get_hotel_data_as_dict method as HotelData class")

            return input_data

        except Exception as e:
            raise HotelException(e, sys) from e

class HotelClassifier:
    def __init__(self,prediction_pipeline_config: HotelPredictorConfig = HotelPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise HotelException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of Hotel Booking Classifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of HotelClassifier class")
            model = HotelEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise HotelException(e, sys)