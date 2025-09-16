import logging
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

from flask import Flask, request, render_template, redirect, url_for, session
import logging

app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for session

## route for homepage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            # Only show result if it exists in session
            results = session.pop("results", None)  
            return render_template('home.html', results=results)
            # return render_template('home.html')
        else:
            # Collect all form inputs
            data = CustomData(
                anxiety_level=request.form.get('anxiety_level'),
                self_esteem=request.form.get('self_esteem'),
                mental_health_history=request.form.get('mental_health_history'),
                depression=request.form.get('depression'),
                headache=request.form.get('headache'),
                blood_pressure=request.form.get('blood_pressure'),
                sleep_quality=request.form.get('sleep_quality'),
                breathing_problem=request.form.get('breathing_problem'),
                noise_level=request.form.get('noise_level'),
                living_conditions=request.form.get('living_conditions'),
                safety=request.form.get('safety'),
                basic_needs=request.form.get('basic_needs'),
                academic_performance=request.form.get('academic_performance'),
                study_load=request.form.get('study_load'),
                teacher_student_relationship=request.form.get('teacher_student_relationship'),
                future_career_concerns=request.form.get('future_career_concerns'),
                social_support=request.form.get('social_support'),
                peer_pressure=request.form.get('peer_pressure'),
                extracurricular_activities=request.form.get('extracurricular_activities'),
                bullying=request.form.get('bullying')
            )

            # Convert to dataframe
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            # Run prediction
            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)[0]
            print("After Prediction")

            # Return prediction result
            print(results)
            logging.info(results)
            if results<0.66:
                output="low"
            elif 0.66<results<1.32:
                output="Moderate"
            else :
                output="High"
                
            # store result in session temporarily
            session["results"] = output

            # Redirect so refresh doesn’t re-post
            return redirect(url_for("predict_datapoint"))
                
            # return render_template('home.html', results=output)

    except Exception as e:
        logging.info(f"Error during prediction: {e}")
        return render_template('home.html', results="Error: Invalid input")

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", debug=True)
