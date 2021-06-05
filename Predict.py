import pandas
import joblib
ds=pandas.read_csv('SalaryData.csv')
x=ds['YearsExperience'].values.reshape(30,1)
y=ds['Salary']
from sklearn.linear_model import LinearRegression
mind=LinearRegression()
model=mind.fit(x, y)
print('''    -----------------------------------------------   
             *****    Welcome to the Prediction App    *****
             -----------------------------------------------  \n ''')
while(True):
  s=float(input("Please type year of Experience : "))
  print("Predicted Salary : " , model.predict([[s]]))
  p=input("\nFor Exit press q or for continue click on Enter : ")
  if p =="q":
    break
joblib.dump(model, 'Salary.pk1')

 

