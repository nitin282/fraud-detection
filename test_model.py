import joblib 
model=joblib.load("fraud_model.joblib")
sample_Data=[[500,4,8,50,0,0]]
prediction=model.predict(sample_Data)
print(prediction)