import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try: 
            model_path='artifacts\model.pkl'
            scaler=OneHotEncoder()
            # preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            # preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=scaler.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    
class CustomData:
    def __init__(self,
                anxiety_level:int,
                self_esteem:int,
                mental_health_history:int,
                depression:int,
                headache:int,
                blood_pressure:int,
                sleep_quality:int,
                breathing_problem:int,
                noise_level:int,
                living_conditions:int,
                safety:int,
                basic_needs:int,
                academic_performance:int,
                study_load:int,
                teacher_student_relationship:int,
                future_career_concerns:int,
                social_support:int,
                peer_pressure:int,
                extracurricular_activities:int,
                bullying:int,
                ):
        self.anxiety_level=anxiety_level
        self.self_esteem=self_esteem    
        self.mental_health_history=mental_health_history    
        self.depression=depression 
        self.headache=headache
        self.blood_pressure=blood_pressure
        self.sleep_quality=sleep_quality
        self.breathing_problem=breathing_problem
        self.noise_level=noise_level  
        self.living_conditions=living_conditions
        self.safety=safety 
        self.basic_needs=basic_needs   
        self.academic_performance=academic_performance  
        self.study_load=study_load    
        self.teacher_student_relationship=teacher_student_relationship   
        self.future_career_concerns=future_career_concerns   
        self.social_support=social_support  
        self.peer_pressure=peer_pressure   
        self.extracurricular_activities=extracurricular_activities
        self.bullying=bullying  
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "anxiety_level":[self.anxiety_level],
                "self_esteem":[self.self_esteem],
                "mental_health_history":[self.mental_health_history],
                "depression":[self.depression],
                "headache":[self.headache],
                "blood_pressure":[self.blood_pressure],
                "sleep_quality":[self.sleep_quality],
                "breathing_problem":[self.breathing_problem],
                "noise_level":[self.noise_level],
                "living_conditions":[self.living_conditions],
                "safety":[self.safety],
                "basic_needs":[self.basic_needs],
                "academic_performance":[self.academic_performance],
                "study_load":[self.study_load],
                "teacher_student_relationship":[self.teacher_student_relationship],
                "future_career_concerns":[self.future_career_concerns],
                "social_support":[self.social_support],
                "peer_pressure":[self.peer_pressure],
                "extracurricular_activities":[self.extracurricular_activities],
                "bullying":[self.bullying],
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)