import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide")

# Load dataset
df = pd.read_csv("Ice_cream selling data.csv")
X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# Initialize session state
if 'x_points' not in st.session_state:
    st.session_state.x_points = X
    st.session_state.y_points = y
if 'degree' not in st.session_state:
    st.session_state.degree = 2

# Sidebar controls
st.sidebar.title("Controls")

col1, col2 = st.sidebar.columns([1, 1])
with col1:
    if st.button("➖", key="decrease"):
        if st.session_state.degree > 1:
            st.session_state.degree -= 1
with col2:
    if st.button("➕", key="increase"):
        if st.session_state.degree < 15:
            st.session_state.degree += 1

st.sidebar.markdown(f"**Polynomial Degree: {st.session_state.degree}**")

# Polynomial regression model
def get_polynomial_model(x, y, degree):
    x = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    return model, poly, y_pred, mse

# Compute model
degree = st.session_state.degree
model, poly, y_pred, mse = get_polynomial_model(
    np.array(st.session_state.x_points),
    np.array(st.session_state.y_points),
    degree
)

# Generate fitted line for plotting
x_line = np.linspace(min(X), max(X), 300).reshape(-1, 1)
y_line = model.predict(poly.transform(x_line))

# Create Plotly figure
fig = go.Figure()

# Data points (editable in the table below)
fig.add_trace(go.Scatter(
    x=st.session_state.x_points,
    y=st.session_state.y_points,
    mode='markers',
    name='Data Points',
    marker=dict(size=10, color='blue'),
    text=[f"{i}" for i in range(len(X))],
    customdata=list(range(len(X))),
    hoverinfo="text",
))

# Polynomial regression line
fig.add_trace(go.Scatter(
    x=x_line.flatten(),
    y=y_line,
    mode='lines',
    name='Polynomial Fit',
    line=dict(color='red', width=2)
))

fig.update_layout(
    title=f"Polynomial Regression - Degree {degree} | MSE: {mse:.2f}",
    xaxis_title="Temperature (°C)",
    yaxis_title="Ice Cream Sales (units)",
    dragmode="lasso"
)

# Show plot
st.plotly_chart(fig, use_container_width=True)

# Editable data table
edited_df = st.data_editor(
    pd.DataFrame({
        "Temperature (°C)": st.session_state.x_points,
        "Ice Cream Sales (units)": st.session_state.y_points
    }),
    num_rows="fixed",
    use_container_width=True
)

# Update data points
st.session_state.x_points = edited_df["Temperature (°C)"].values
st.session_state.y_points = edited_df["Ice Cream Sales (units)"].values

# Optional instructions
st.markdown("### Instructions")
st.markdown(
    """
    - Use ➖ / ➕ buttons in the sidebar to change polynomial degree.
    - Edit values in the table below to simulate new data points.
    - Graph and MSE update automatically.
    """
)
