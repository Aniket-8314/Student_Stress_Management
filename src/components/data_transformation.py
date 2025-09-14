import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
    
@dataclass
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            
            target_column_name="stress_level"
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            

            logging.info(
                f"Applying preprocessing object on training and testing dataframe"
            )
            
            preprocessor=StandardScaler()
            train_features = preprocessor.fit_transform(input_feature_train_df)
            test_features = preprocessor.transform(input_feature_test_df)

            # ✅ Add target back
            train_arr = np.c_[train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[test_features, np.array(target_feature_test_df)]
            # train_arr = preprocessor.fit_transform(input_feature_train_df)
            # test_arr = preprocessor.transform(input_feature_test_df)
            
            print("Preprocessor learned features:", preprocessor.feature_names_in_)
            print("Count:", len(preprocessor.feature_names_in_))           


            logging.info(
                f"Applying preprocessing object on training and testing dataframe"
            )
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys) 
        