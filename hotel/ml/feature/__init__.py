import sys

from sklearn.base import BaseEstimator,TransformerMixin
from hotel.exception import HotelException
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
class CutsomFeatureHandler(BaseEstimator, TransformerMixin):

    def __init__(self,):
        self.columns = ["children", "adults", "babies", "lead_time",
                        "meal", "market_segment", "distribution_channel"]

    def __validate_required_column(self,x):
        missing_columns = []
        for column in self.columns:
            if column not in x.columns:
                missing_columns.append(column)

        if len(missing_columns) > 0:
            raise Exception(f"Missing required column : [{missing_columns}]")

        self.ordinal_obj=None
        self.standard_scaler = None


    def fit(self, x:DataFrame, y=None):
        # self.__validate_required_column(x)
        self.standard_scaler = None
        return self

    def transform(self, x, y=None):

        try:
            # self.__validate_required_column(x)
            x["total_guest"] = x['children'] + x["adults"] + x["babies"]

            x.drop(["babies","adults","children"],axis=1,inplace=True)

            x["lead_time"] = (x["lead_time"] / 24).round(2)

            # replace meal Undefined with Self Catering
            x["meal"].replace("Undefined", "SC", inplace=True)
            # Replace
            x["market_segment"].replace("Undefined", "Online TA", inplace=True)
            # x.drop(x[x['distribution_channel'] == 'Undefined'].index, inplace=True, axis=0)

            if self.standard_scaler is None:
                sc = StandardScaler()
                self.standard_scaler =  sc.fit( x[["total_guest"]])
            x["total_guest"]  =self.standard_scaler.transform( x[["total_guest"]])
            print(x.head())
            return x

        except Exception as e:
            raise HotelException(e, sys) from e
