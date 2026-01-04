import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#page configuration
st.set_page_config(page_title = "Multiple Linear Regression App", layout = "centered")

#load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
load_css("style.css")

#title
st.markdown("""
            <div class = "card">
            <h1>Multiple Linear Regression App</h1>
            <p> Predict<b> Tip Amount </b> from <b> Multiple Features </b> using Multiple Linear Regression</p>
            </div>
""", unsafe_allow_html = True)

#load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df =  load_data()

#dataset preview
st.markdown('<div class = "card"><h2>Dataset Preview</h2></div>', unsafe_allow_html = True)
st.dataframe(df.head())

#prepare data - encode categorical variables
df_encoded = df.copy()
df_encoded = pd.get_dummies(df_encoded, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)

#select features
feature_cols = [col for col in df_encoded.columns if col != 'tip']
x = df_encoded[feature_cols]
y = df_encoded['tip']

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#train model
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

#model evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - x.shape[1] - 1)

#feature importance visualization
st.markdown('<div class = "card"><h2>Feature Importance (Coefficients)</h2></div>', unsafe_allow_html = True)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance['Feature'], feature_importance['Coefficient'])
ax.set_xlabel('Coefficient Value')
ax.set_ylabel('Feature')
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
st.pyplot(fig)

#actual vs predicted
st.markdown('<div class = "card"><h2>Actual vs Predicted Tips</h2></div>', unsafe_allow_html = True)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha = 0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual Tip")
ax.set_ylabel("Predicted Tip")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

#performance
st.markdown('<div class = "card"><h2>Model Performance</h2>', unsafe_allow_html = True)
c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3, c4 = st.columns(2)
c3.metric("R2", f"{r2:.2f}")
c4.metric("Adjusted R2", f"{adj_r2:.2f}")
st.markdown('</div>', unsafe_allow_html = True)

#model intercept
st.markdown(f"""
            <div class = "card">
            <h3>Model Intercept</h3>
            <p> <b> Intercept : </b> {model.intercept_:.3f} </p>
            </div>
""", unsafe_allow_html = True)

#prediction section
st.markdown('<div class = "card"><h2>Make a Prediction</h2></div>', unsafe_allow_html = True)

col1, col2 = st.columns(2)
with col1:
    bill = st.slider("Total Bill ($)", float(df["total_bill"].min()), float(df["total_bill"].max()), 20.0)
    size = st.slider("Party Size", int(df["size"].min()), int(df["size"].max()), 2)
    sex = st.selectbox("Sex", ["Female", "Male"])

with col2:
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
    time = st.selectbox("Time", ["Lunch", "Dinner"])

#prepare input for prediction
input_data = pd.DataFrame({
    'total_bill': [bill],
    'size': [size],
    'sex': [sex],
    'smoker': [smoker],
    'day': [day],
    'time': [time]
})

# Apply same encoding as training data
input_encoded = pd.get_dummies(input_data, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)

# Ensure all columns match training data
for col in feature_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns to match training data
input_encoded = input_encoded[feature_cols]
input_scaled = scaler.transform(input_encoded)
tip = model.predict(input_scaled)[0]

st.markdown(f'''
            <div class = "card">
                <h2>Predicted Tip Amount: $ {tip:.2f}</h2>
            </div>
            ''', unsafe_allow_html = True)
