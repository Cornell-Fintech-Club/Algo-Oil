import pandas as pd
import prophet
from matplotlib import pyplot

data = pd.read_csv('eia-data/Weekly_U.S._Field_Production_of_Crude_Oil.csv')
regressor = pd.read_csv('eia-data/Weekly_U._S._Operable_Crude_Oil_Distillation_Capacity.csv')

#  fix column names for prophet
data.columns = ['ds', 'y']
data['ds']= pd.to_datetime(data['ds'])

regressor.columns = ['ds', 'Capacity']
regressor['ds']= pd.to_datetime(regressor['ds'])

data = data.merge(regressor, on='ds')

model = prophet.Prophet()

model.fit(data)

future = []
for i in range(1, 13): #number of months
    future.append(['2024-%02d' % i  ])
future = pd.DataFrame(future, columns=['ds'])
future['ds']= pd.to_datetime(future['ds'])

# use the model to make a forecast
forecast = model.predict(future)

# prints numerical info
# print(forecast[['ds', 'yhat']].head())

model.plot(forecast)
pyplot.show()