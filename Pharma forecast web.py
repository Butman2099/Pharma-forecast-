import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import calendar
import math

# Function to apply trend adjustment
def inverse_transform(prediction, trend):
    return prediction + trend

# Set page configuration
st.set_page_config(
    page_title="Інструмент прогнозування фармацевтичних продажів",
    page_icon="📊",
    layout="wide",
)

# Application title and description
st.title("Інструмент прогнозування фармацевтичних продажів")
st.markdown("""
Ця програма дозволяє завантажити модель машинного навчання і дані про продажі, 
щоб спрогнозувати продажі фармацевтичної компанії, вибравши конкретну дату прогнозу.
""")

# Sidebar for model upload and configuration
with st.sidebar:
    st.header("Завантаження моделі")
    uploaded_model = st.file_uploader("Завантажте вашу модель ML (.pkl або .joblib)", type=["pkl", "joblib"])
    
    st.header("Конфігурація даних")
    uploaded_data = st.file_uploader("Завантажте дані по продажам (CSV)", type=["csv"])

# Initialize session state for model and data
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = [
        "Remote communication", "E-mailing", "Press advertisement", 
        "Radio advertisement", "P_UAH_SHIFT_1"
    ]

# Load model when uploaded
if uploaded_model is not None:
    try:
        model_bytes = io.BytesIO(uploaded_model.read())
        st.session_state.model = joblib.load(model_bytes)
        st.sidebar.success("Модель успішно завантажено!")
    except Exception as e:
        st.sidebar.error(f"Помилка із завантаженням моделі: {e}")

# Load data if uploaded
if uploaded_data is not None:
    try:
        st.session_state.data = pd.read_csv(uploaded_data)
        st.sidebar.success("Дані успішно завантажено!")
        
        # Convert DATE column to datetime
        if 'DATE' in st.session_state.data.columns:
            st.session_state.data['DATE'] = pd.to_datetime(st.session_state.data['DATE'])
        
        # Extract features from data (excluding DATE and Time index)
        if st.session_state.data is not None:
            st.session_state.features = [col for col in st.session_state.data.columns 
                                      if col not in ['DATE', 'Time index']]
    except Exception as e:
        st.sidebar.error(f"Помилка із завантаженням даних: {e}")

# Display sample data and feature statistics if available
if st.session_state.data is not None:
    st.header("Вибірка даних")
    st.dataframe(st.session_state.data.head())

# Main prediction section
st.header("Створення прогнозу")

# Date selection for forecast
if st.session_state.data is not None and 'DATE' in st.session_state.data.columns:
    # Get the range of dates available in the data
    min_date = st.session_state.data['DATE'].min()
    max_date = st.session_state.data['DATE'].max()
    
    # Function to add one month to a date
    def add_one_month(dt):
        month = dt.month
        year = dt.year
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1
        
        # Handle edge cases with month lengths
        last_day = calendar.monthrange(year, month)[1]
        day = min(dt.day, last_day)
        
        return dt.replace(year=year, month=month, day=day)
    
    # Calculate next month after the last available date for max forecast date
    max_forecast_date = add_one_month(max_date)
    
    # Get min forecast date (should be one month after the first date in the dataset)
    min_forecast_date = add_one_month(min_date)
    
    # Date picker for forecast date
    forecast_date = st.date_input(
        "Виберіть дату прогнозу:",
        value=min_forecast_date,
        min_value=min_forecast_date,
        max_value=max_forecast_date
    )
    
    # Convert to datetime for comparison
    forecast_date = pd.to_datetime(forecast_date)
    
    # Calculate the input date (1 month before forecast date)
    input_date = forecast_date - pd.DateOffset(months=1)
    input_date = pd.to_datetime(input_date).normalize()
    
    # Find the closest date in the dataset
    if st.session_state.data is not None:
        closest_date_mask = st.session_state.data['DATE'] == input_date
        
        if closest_date_mask.any():
            st.success(f"Використовуємо дані з: {input_date.strftime('%Y-%m-%d')} для прогнозування продажів на: {forecast_date.strftime('%Y-%m-%d')}")
            input_row = st.session_state.data[closest_date_mask].iloc[0]
            
            # Display the input data for the selected date
            input_features = {}
            
            # Option to use CSV values or enter custom values
            use_csv_values = st.radio(
                "Вхідне джерело даних:",
                ["Використати значення із CSV", "Вввести значення до полів"],
                horizontal=True
            )
            
            cols = st.columns(2)
            
            for i, feature in enumerate(st.session_state.features):
                with cols[i % 2]:
                    # Determine appropriate decimal places based on the value
                    value = input_row[feature]
                    #if abs(value) < 1:
                        #decimal_places = 3
                    #elif abs(value) < 10:
                        #decimal_places = 2
                    #elif abs(value) < 100:
                        #decimal_places = 1
                    #else:
                        #decimal_places = 0
                    
                    # Create number input field (disabled if using CSV values)
                    input_features[feature] = st.number_input(
                        f"{feature}",
                        value=value,
                        #format=f"%.{decimal_places}f",
                        disabled=(use_csv_values == "Використати значення із CSV")
                    )
                    
                    # If using CSV values, always use the original value
                    if use_csv_values == "Використати значення із CSV":
                        input_features[feature] = value
            
            # Make prediction button
            if st.button("Генерація прогнозу"):
                if st.session_state.model is not None:
                    try:
                        # Create input dataframe
                        input_df = pd.DataFrame([input_features])
                        
                        # Make prediction
                        prediction = st.session_state.model.predict(input_df)
                        
                        # Get Time index for trend calculation
                        time_index = None
                        if 'Time index' in input_row:
                            time_index = input_row['Time index']
                            
                            # Calculate trend adjustment
                            st.subheader("Прогноз із поправкою на тренд")
                            
                            # Applying logarithmic trend formula: y = 1307.0931 + 315.7310*log(x)
                            trend_value = 1307.0931 + 315.7310 * np.log(time_index)
                            
                            # For forecast month (time_index + 1)
                            forecast_time_index = time_index + 1
                            forecast_trend_value = 1307.0931 + 315.7310 * np.log(forecast_time_index)
                            
                            # Adjust prediction with trend using inverse_transform function
                            adjusted_prediction = inverse_transform(prediction[0], forecast_trend_value)
                            
                            # Display both predictions
                            col = st.columns(1)[0]
                          
                            with col:
                                st.metric(
                                    label=f"Прогноз із поправкою на тренд",
                                    value=f"{adjusted_prediction:.4f}"
                                )
                            
                            # Show trend calculation details
                            with st.expander("Показати деталі розрахунку"):
                                st.write(f"Формула логарифмічного тренду: y = 1307.0931 + 315.7310 * log(x)")
                                st.write(f"Часовий індекс вхідного місяця ({input_date.strftime('%Y-%m-%d')}): {time_index}")
                                for idx, column in enumerate(input_df.columns):
                                    value = input_df.iloc[0, idx]
                                    st.write(f"{column} = {value}")
                        else:
                            # Display prediction without trend adjustment
                            st.header("Прогнозні результати")
                            st.markdown(f"### Прогнозовані продажі для {forecast_date.strftime('%Y-%m-%d')}: **{prediction[0]:.4f}**")
                            st.warning("Неможливо застосувати поправку на тренд, оскільки колонка «Time index» відсутня у даних.")
                        
                        # Visualize the prediction
                        st.subheader("Використані фактори")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Create simple bar chart of input features
                        feature_names = list(input_features.keys())
                        feature_values = list(input_features.values())
                        
                        ax.bar(feature_names, feature_values)
                        ax.set_xticklabels(feature_names, rotation=45, ha='right')
                        ax.set_title(f"Вхідні значення факторів для {input_date.strftime('%Y-%m-%d')}")
                        ax.set_ylabel("Значення")
                        
                        st.pyplot(fig)
                        
                        # Additional feature importance if available
                        if hasattr(st.session_state.model, 'feature_importances_'):
                            st.subheader("Feature Importance")
                            importances = st.session_state.model.feature_importances_
                            
                            fig2, ax2 = plt.subplots(figsize=(10, 5))
                            sorted_idx = np.argsort(importances)
                            ax2.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
                            ax2.set_title("Feature Importance")
                            
                            st.pyplot(fig2)
                            
                    except Exception as e:
                        st.error(f"Помилка при створені прогнозу: {e}")
                else:
                    st.warning("Спочатку завантажте модель.")
        else:
            st.error(f"Дані не доступні для {input_date.strftime('%Y-%m-%d')}. Виберіть іншу прогнозну дату.")
else:
    st.warning("Будь ласка, завантажуйте дані з колонкою DATE в першу чергу.")

# Display trend over time if data available
if st.session_state.data is not None and 'DATE' in st.session_state.data.columns:
    st.header("Історичні значення показників")
    
    # Plot time series for features with normalized values
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a copy of the dataframe for normalization
    plot_data = st.session_state.data.copy()
    
    for feature in st.session_state.features:
        if feature != 'DATE':
            # Min-max normalization to scale features between 0 and 1
            min_val = plot_data[feature].min()
            max_val = plot_data[feature].max()
            # Avoid division by zero
            if max_val > min_val:
                plot_data[f"{feature}_normalized"] = (plot_data[feature] - min_val) / (max_val - min_val)
            else:
                plot_data[f"{feature}_normalized"] = plot_data[feature] / max_val if max_val != 0 else plot_data[feature]
            
            # Plot normalized values
            ax.plot(plot_data['DATE'], plot_data[f"{feature}_normalized"], label=feature)
    
    ax.set_title("Динаміка факторів із плином часу (Нормалізований масштаб)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Нормалізовані значення (0-1)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

# About section
st.sidebar.header("About")
st.sidebar.info("""
Ця програма дозволяє прогнозувати фармацевтичні продажі на місяць вперед.
Завантажте свою модель і дані про продажі, оберіть дату прогнозу і отримайте прогноз миттєво.
Модель використовує дані за місяць до прогнозу + значення цін на поточний прогнозний місяць.
""")

# Current date information
current_date = datetime.now().strftime("%Y-%m-%d")
st.sidebar.markdown(f"**Поточна дата:** {current_date}")