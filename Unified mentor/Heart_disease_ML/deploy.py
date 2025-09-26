import joblib
import pandas as pd
import numpy as np


def p_h_d():
	try:
		model = joblib.load("/home/shobhit/AI-ML/Unified mentor/Heart_disease_ML/data cum deploy/heart_disease_model.joblib")
		scaler = joblib.load("/home/shobhit/AI-ML/Unified mentor/Heart_disease_ML/data cum deploy/scaler.joblib")
		
		
	except FileNotFoundError:
		print("Model is not found please check the specified location of the file\n")
		return
	
	feature_names  = [
		"age", "sex", "chest pain type", "resting bp s", "cholesterol",
		"fasting blood sugar", "resting ecg", "max heart rate",
		"exercise angina", "oldpeak", "ST slope"
	]

	user_input = {}
	for feature in feature_names:
		while True:
			try:
				value=float(input(f"Enter the Value for {feature}: "))
				user_input[feature]=value
				break
			except ValueError:
				print("Invalid Input. please enter a numerical value. ")
		input_df=pd.DataFrame([user_input])

		input_df = input_df[feature_names]
		prediction = model.predict(input_df)
		
		if prediction[0] == 1:
			print("\n Most probably the user has a heart disease: ")
		else:
			print("\n the user is most probably fine")
	
if 	__name__ == "__main__":
	p_h_d()
