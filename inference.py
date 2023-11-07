import joblib
import pandas as pd

model = joblib.load("./model/linear_model.joblib")

pregnancies = int(input("Enter Pregnancies: "))
glucose = int(input("Enter Glucose: "))
bloodpressure = int(input("Enter the Blood Pressure: "))
skinthickness = int(input("Enter the thickness: "))
insulin = int(input("Enter the Insulin: "))
bmi = float(input("Enter the BMI: "))
diabetesPedigreefunction = float(input("Enter the PedigreeFunction: "))
age = int(input("Enter the age: "))

user_input = [[pregnancies, glucose, bloodpressure,skinthickness,insulin,bmi,diabetesPedigreefunction,age]]

predictions = model.predict(user_input)
classes = ["Positive","Negative"]
print("Predicted class is",classes[int(predictions[0])])

print("Everything done")
