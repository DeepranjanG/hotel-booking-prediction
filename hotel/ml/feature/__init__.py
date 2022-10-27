import sys
from hotel.exception import HotelException
from hotel.logger import logging
from pandas import DataFrame

def basic_preprocessing(dataframe: DataFrame):

    """

    Parameters
    ----------
    dataframe: Take dataframe as input

    Returns: DataFrame after applying basic preprocessing techniques
    -------

    """
    try:

        logging.info("Creating new total_guest column using three column names")

        dataframe["total_guest"] = dataframe['children'] + dataframe["adults"] + dataframe["babies"]

        logging.info("Dropping merged columns")

        dataframe.drop(["babies", "adults", "children"], axis=1, inplace=True)

        logging.info("Converting lead_time value. Hours into days")

        dataframe["lead_time"] = ((dataframe["lead_time"].astype(int)) / 24).round(2)

        logging.info("Replacing Undefined values")

        dataframe["meal"].replace("Undefined", "SC", inplace=True)

        dataframe["market_segment"].replace("Undefined", "Online TA", inplace=True)

        logging.info("Dropping row where value is Undefined")

        dataframe.drop(dataframe[dataframe['distribution_channel'] == 'Undefined'].index, inplace=True, axis=0)

    except Exception as e:
        raise HotelException(e, sys) from e

    return dataframe