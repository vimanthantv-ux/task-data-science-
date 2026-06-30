import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import pandas as pd
data=pd.read_csv("icecream_sales.csv")
data

A=data["Temperature"]
A

scales=data["IceCreamSales"]
scales

model=LinearRegression()

x=np.array(A).reshape(-1,1)
y=np.array(scales).reshape(-1,1)

x

y

model.fit(x,y)

new_Tempreature=int(input("enter your tempreature"))
new_data=np.array(new_Tempreature).reshape(-1,1)
y_answer=model.predict(new_data)
print("your icecream scales",model.predict(new_data))

plt.plot(x,y)
plt.scatter(new_data,y_answer)
plt.xlabel("Tempreature")
plt.ylabel("icecream scales")
plt.title("tempreature vs icecream scales")
plt.show()



