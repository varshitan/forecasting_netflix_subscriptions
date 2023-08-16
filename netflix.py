import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set the Plotly default template
pio.templates.default = "plotly_white"

# Read the data
data = pd.read_csv('Netflix-Subscriptions.csv')
data['Time Period'] = pd.to_datetime(data['Time Period'], format='%d/%m/%Y')

# Create a line plot of Netflix subscriptions over time
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Time Period'],
                         y=data['Subscribers'],
                         mode='lines', name='Subscribers'))
fig.update_layout(title='Netflix Quarterly Subscriptions Growth',
                  xaxis_title='Date',
                  yaxis_title='Netflix Subscriptions')
fig.show()

# Calculate the quarterly growth rate
data['Quarterly Growth Rate'] = data['Subscribers'].pct_change() * 100

# Extract the year from the 'Time Period' column
data['Year'] = data['Time Period'].dt.year

# Create a new column for bar color (green for positive growth, red for negative growth)
data['Bar Color'] = data['Quarterly Growth Rate'].apply(lambda x: 'green' if x > 0 else 'red')

# Plot the yearly growth rate using bar graphs
yearly_growth = data.groupby('Year')['Quarterly Growth Rate'].sum()
fig = go.Figure()
fig.add_trace(go.Bar(
    x=yearly_growth.index,
    y=yearly_growth,
    marker_color=data['Bar Color'],
    name='Yearly Growth Rate'
))
fig.update_layout(title='Netflix Yearly Subscriber Growth Rate',
                  xaxis_title='Year',
                  yaxis_title='Yearly Growth Rate (%)')
fig.show()

# Create a time series and a differenced series
time_series = data.set_index('Time Period')['Subscribers']
differenced_series = time_series.diff().dropna()

# Plot ACF and PACF of differenced time series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()

# Find the best p, d, and q values using auto_arima
model = auto_arima(time_series, seasonal=False, stepwise=True, trace=True)
best_p, best_d, best_q = model.order

# Fit the ARIMA model with the best parameters
best_model = ARIMA(time_series, order=(best_p, best_d, best_q))
results = best_model.fit()
print(results.summary())

# Make predictions
future_steps = 5
predictions = results.predict(start=len(time_series), end=len(time_series) + future_steps - 1)
predictions = predictions.astype(int)

# Create a DataFrame with the original data and predictions
forecast = pd.DataFrame({'Original': time_series, 'Predictions': predictions})

# Plot the original data and predictions
fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Predictions'],
                         mode='lines', name='Predictions'))

fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Original'],
                         mode='lines', name='Original Data'))

fig.update_layout(title='Netflix Quarterly Subscription Predictions',
                  xaxis_title='Time Period',
                  yaxis_title='Subscribers',
                  legend=dict(x=0.1, y=0.9),
                  showlegend=True)

fig.show()
