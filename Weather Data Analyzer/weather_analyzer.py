import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("weather.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df.fillna(df.mean(numeric_only=True), inplace=True)

yearly_temp = df.groupby('Year')['Temperature'].mean()
yearly_rain = df.groupby('Year')['Rainfall'].sum()

fig, axs = plt.subplots(2, 2, figsize=(14, 8))
fig.tight_layout(pad=4.0)

axs[0, 0].plot(yearly_temp.index, yearly_temp.values, color='red', marker='o', label='Temperature (°C)')
axs[0, 0].set_title('Temperature Trends Over Years')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Temperature (°C)')
axs[0, 0].legend()
axs[0, 0].grid()

axs[0, 1].bar(yearly_rain.index, yearly_rain.values, color='blue', label='Rainfall (mm)')
axs[0, 1].set_title('Yearly Rainfall Distribution')
axs[0, 1].set_xlabel('Year')
axs[0, 1].set_ylabel('Rainfall (mm)')
axs[0, 1].legend()

axs[1, 0].scatter(df['Temperature'], df['Humidity'], color='green', marker='x')
axs[1, 0].set_title('Humidity vs Temperature Correlation')
axs[1, 0].set_xlabel('Temperature (°C)')
axs[1, 0].set_ylabel('Humidity (%)')

X = df[['Year']]
y = df['Temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X)

axs[1, 1].scatter(X, y, color='purple', label='Actual Temperature', s=10)
axs[1, 1].plot(X, predictions, color='orange', linestyle='--', label='Predicted Trend')
axs[1, 1].set_title('Temperature Prediction for Next Years')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Temperature (°C)')
axs[1, 1].legend()

plt.show()
