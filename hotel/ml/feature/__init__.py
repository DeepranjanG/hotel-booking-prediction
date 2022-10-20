import sys
from hotel.exception import HotelException
from pandas import DataFrame

def basic_preprocessing(dataframe: DataFrame):
    try:

        dataframe["total_guest"] = dataframe['children'] + dataframe["adults"] + dataframe["babies"]

        dataframe.drop(["babies", "adults", "children"], axis=1, inplace=True)

        dataframe["lead_time"] = ((dataframe["lead_time"].astype(int)) / 24).round(2)

        dataframe["meal"].replace("Undefined", "SC", inplace=True)

        dataframe["market_segment"].replace("Undefined", "Online TA", inplace=True)

        dataframe.drop(dataframe[dataframe['distribution_channel'] == 'Undefined'].index, inplace=True, axis=0)

    except Exception as e:
        raise HotelException(e, sys) from e

    return dataframe