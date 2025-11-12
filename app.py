import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
from huggingface_hub import hf_hub_download, login
import os
import re
import requests
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="AgroAssist", layout="wide")

#st.sidebar.title("🌾 Smart Crop Care")
# Sidebar navigation
st.sidebar.markdown("<h1 style='color:#2e7d32;'>🌾 Smart Crop Care</h1>", unsafe_allow_html=True)
st.sidebar.markdown("### Select a Feature")
page = st.sidebar.radio(
    "Choose an option",
    ["🌿 Disease Detection", "☀️ Weather Forecast", "💬 Chatbot"]
)


if page == "🌿 Disease Detection":
    # Load fertilizer recommendations dataset
    disease_df = pd.read_csv("disease_dataset.csv")


    # Load model
    # Cache model load to avoid reloading on every user action
    @st.cache_resource
    def load_model():
        hf_token = st.secrets.get("HUGGINGFACE_TOKEN", os.getenv("HUGGINGFACE_TOKEN"))

        if hf_token:
            login(token=hf_token)
        else:
            st.error("⚠️    Missing Hugging Face token. Please set HUGGINGFACE_TOKEN in secrets.toml.")

        REPO_ID = "JenishMakwana/plant_disease_resnet50"
        FILENAME = "best_resnet_model_224x224.keras"

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            token=hf_token
        )

        model = tf.keras.models.load_model(model_path, compile=False)
        return model

    # ✅ Model is cached after first load
    model = load_model()
    
    # Alternative: Load model from local file if available
    #model = tf.keras.models.load_model("best_resnet_model_224x224.keras", compile=False)

    # Load class names (must be saved from notebook)
    class_names = np.load("class_names.npy", allow_pickle=True)

    # Preprocess function
    def preprocess_image(img, target_size=(224, 224)):
        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
        return img_array

    # Prediction function
    def predict_image(model, img, class_names):
        img_array = preprocess_image(img)
        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds,axis=1)[0]]
        confidence = np.max(preds)
        return pred_class, confidence

    # Fertilizer recommendation function
    def get_disease_info(disease_name):
        row = disease_df[disease_df['Disease_Name'].str.lower() == disease_name.lower()]
        if not row.empty:
            info = row.iloc[0]
            return {
                "Category": info["Category"],
                "Fertilizers": info["Fertilizers"],
                "Eco-Friendly Treatment": info["Eco_Friendly_Treatment"],
                "Chemical Fungicide/Bactericide": info["Chemical_Fungicide_Bactericide"],
                "Notes": info["Notes"]
            }
        else:
            return None



    # Streamlit UI

    st.title("🌱 Plant Disease Detection")
    st.write("Upload a leaf image to predict the disease category.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            img = img.resize((300, 300))
            st.image(img, caption="Uploaded Image")

        # Prediction
        with st.spinner("Predicting..."):
            pred_class, confidence = predict_image(model, img, class_names)
        st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #f1f8e9 0%, #e8f5e9 100%);
                border-left: 6px solid #4caf50;
                padding: 20px;
                border-radius: 12px;
                margin-top: 20px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                <h3><b style="color:#1b5e20;">Prediction:</b> {pred_class}</h3>
            </div>
            """, unsafe_allow_html=True)

        #st.info(f"**Confidence:** {confidence:.2f}")
        # Get fertilizer recommendation
        # Fetch recommendation data
        info = get_disease_info(pred_class)

        if info:
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #f1f8e9 0%, #e8f5e9 100%);
                border-left: 6px solid #4caf50;
                padding: 20px;
                border-radius: 12px;
                margin-top: 20px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                <h3 style="color:#2e7d32;">🌿 Disease Treatment & Fertilizer Recommendation</h3>
                <p><b style="color:#1b5e20;">Category:</b> {info['Category']}</p>
                <p><b style="color:#1b5e20;">Fertilizers:</b> {info['Fertilizers']}</p>
                <p><b style="color:#1b5e20;">Eco-Friendly Treatment:</b> {info['Eco-Friendly Treatment']}</p>
                <p><b style="color:#1b5e20;">Chemical Fungicide/Bactericide:</b> {info['Chemical Fungicide/Bactericide']}</p>
                <p><b style="color:#1b5e20;">Notes:</b> {info['Notes']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No treatment data found for this disease.")
            
elif page == "☀️ Weather Forecast":
    import streamlit as st
    import requests
    import pandas as pd
    from PIL import Image
    import base64
    from io import BytesIO
    # Weathercode descriptions from Open-Meteo docs
    WEATHERCODE_DESCRIPTIONS = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }

    # Map weathercode to open-meteo icons (or placeholder emojis)
    WEATHERCODE_ICONS = {
        0: "☀️",
        1: "🌤️",
        2: "⛅",
        3: "☁️",
        45: "🌫️",
        48: "🌫️",
        51: "🌦️",
        53: "🌧️",
        55: "🌧️",
        56: "🌧️❄️",
        57: "🌧️❄️",
        61: "🌧️",
        63: "🌧️",
        65: "🌧️",
        66: "🌧️❄️",
        67: "🌧️❄️",
        71: "❄️",
        73: "❄️",
        75: "❄️",
        77: "❄️",
        80: "🌦️",
        81: "🌧️",
        82: "⛈️",
        85: "🌨️",
        86: "❄️",
        95: "⛈️",
        96: "⛈️",
        99: "⛈️"
    }

    def get_ip_location():
        try:
            res = requests.get("https://ipinfo.io/json")
            data = res.json()
            lat, lon = map(float, data["loc"].split(","))
            city = data.get("city", "")
            return lat, lon, city
        except Exception as e:
            st.error(f"Error detecting location via IP: {e}")
            return None, None, None

    def geocode_city(city_name):
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": city_name, "format": "json", "limit": 1}
            response = requests.get(url, params=params, headers={"User-Agent": "StreamlitApp"})
            data = response.json()
            if data and len(data) > 0:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                return lat, lon
            else:
                st.error(f"Could not find coordinates for city: {city_name}")
                return None, None
        except Exception as e:
            st.error(f"Error in geocoding: {e}")
            return None, None

    def get_weather_by_coords(lat, lon):
        OPENWEATHER_API_KEY = "80338847e882892b9d3fdc04654dd858"  # Replace with your key
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "Location": data["name"],
                "Temperature (°C)": data["main"]["temp"],
                "Humidity (%)": data["main"]["humidity"],
                "Weather": data["weather"][0]["description"].title(),
                "Wind Speed (m/s)": data["wind"]["speed"]
            }
        else:
            st.error(f"OpenWeatherMap API error: {response.status_code} - {response.text}")
            return None

    def get_weather_by_city(city):
        OPENWEATHER_API_KEY = "80338847e882892b9d3fdc04654dd858"  # Replace with your key
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "Location": data["name"],
                "Temperature (°C)": data["main"]["temp"],
                "Humidity (%)": data["main"]["humidity"],
                "Weather": data["weather"][0]["description"].title(),
                "Wind Speed (m/s)": data["wind"]["speed"]
            }
        else:
            st.error(f"OpenWeatherMap API error: {response.status_code} - {response.text}")
            return None

    def get_long_term_forecast(lat, lon):
        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,weathercode",
            "timezone": "auto"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Open-Meteo API error: {response.status_code}")
            return None

    def style_table():
        st.markdown(
            """
            <style>
            .dataframe tbody tr:hover {
                background-color: #f0f0f5 !important;
            }
            table {
                width: 100% !important;
                border-collapse: collapse;
            }
            th, td {
                padding: 10px 15px !important;
                border: 1px solid #ddd !important;
                text-align: center !important;
            }
            thead th {
                background-color:#3057D5 !important;
                color: white !important;
                font-weight: bold !important;
                font-size: 16px !important;
            }
            tbody tr:nth-child(even) {
                background-color: #f9f9f9 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def display_forecast(forecast):
        if not forecast or "daily" not in forecast:
            st.error("No forecast data available.")
            return

        daily = forecast["daily"]
        df = pd.DataFrame({
            "Date": daily["time"],
            "Max Temp (°C)": daily["temperature_2m_max"],
            "Min Temp (°C)": daily["temperature_2m_min"],
            "Precipitation (mm)": daily["precipitation_sum"],
            "Rain (mm)": daily["rain_sum"],
            "Showers (mm)": daily["showers_sum"],
            "Snow (mm)": daily["snowfall_sum"],
            "Weather": [f"{WEATHERCODE_ICONS.get(code, '')} {WEATHERCODE_DESCRIPTIONS.get(code, 'Unknown')}" for code in daily["weathercode"]]
        })

        # Temperature styling: background + dark text
        def temp_color(val):
            if val < 0:
                color = 'lightblue'
                text_color = 'black'
            elif val < 10:
                color = '#a1cfff'
                text_color = 'black'
            elif val < 25:
                color = '#ffeb99'
                text_color = '#3a3a00'
            else:
                color = '#ff9999'
                text_color = 'black'
            return f'background-color: {color}; color: {text_color}'

        # Precipitation styling: background + dark text
        def precip_color(val):
            if val == 0:
                color = 'white'
                text_color = 'black'
            elif val < 1:
                color = '#d0f0fd'
                text_color = '#08306b'
            elif val < 5:
                color = '#a0dafd'
                text_color = '#041f4a'
            else:
                color = '#68b8f8'
                text_color = '#001933'
            return f'background-color: {color}; color: {text_color}'

        styler = (
            df.style
            .applymap(temp_color, subset=['Max Temp (°C)', 'Min Temp (°C)'])
            .applymap(precip_color, subset=['Precipitation (mm)', 'Rain (mm)', 'Showers (mm)', 'Snow (mm)'])
            .set_properties(**{'font-size': '14px', 'text-align': 'center'})
            .set_table_styles([{
                'selector': 'th',
                'props': [('background-color', '#3057D5'),
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('font-size', '16px')]
            }])
        )

        st.subheader("7-Days Weather Forecast")
        st.write(styler, unsafe_allow_html=True)

    def main():
        st.title("🌤️ Realtime Weather Forecast")
        st.markdown(
            """
            Get current weather and 7-days forecast for your location.
            Choose your location input method and see detailed daily weather updates.
            """
        )

        location_option = st.radio(
            "Choose how to provide your location:",
            ("Auto detect by IP", "Enter coordinates manually", "Enter city name"),
        )

        weather = None
        forecast = None
        lat = lon = None

        col1, col2 = st.columns([2, 5])

        with col1:
            st.header("Location Input")

            if location_option == "Auto detect by IP":
                lat, lon, city = get_ip_location()
                if lat and lon:
                    st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #f1f8e9 0%, #e8f5e9 100%);
                    border-left: 6px solid #4caf50;
                    padding: 20px;
                    border-radius: 12px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                    <h5 color:#1b5e20; font-weight:600; margin-bottom:8px;">
                    🌍 Detected Location: {city}
                    </h5>
                    <h5 color:#2e7d32; margin-top:0;">
                    ({lat:.4f}, {lon:.4f})
                    </h5>
                    </div>
                """, unsafe_allow_html=True)
                else:
                    st.error("Could not detect location via IP.")

            elif location_option == "Enter coordinates manually":
                lat_input = st.text_input("Latitude", placeholder="27.4698")
                lon_input = st.text_input("Longitude", placeholder="153.0251")

            else:  # city name input
                city_name = st.text_input("City Name", placeholder="Brisbane")

            if st.button("Get Weather and Forecast"):
                if location_option == "Auto detect by IP":
                    if lat and lon:
                        weather = get_weather_by_coords(lat, lon)
                        forecast = get_long_term_forecast(lat, lon)
                    else:
                        st.error("Unable to get location coordinates.")
                elif location_option == "Enter coordinates manually":
                    try:
                        lat = float(lat_input.strip())
                        lon = float(lon_input.strip())
                        weather = get_weather_by_coords(lat, lon)
                        forecast = get_long_term_forecast(lat, lon)
                    except Exception:
                        st.error("Please enter valid numeric latitude and longitude.")
                else:
                    if city_name.strip():
                        weather = get_weather_by_city(city_name.strip())
                        lat, lon = geocode_city(city_name.strip())
                        if lat and lon:
                            forecast = get_long_term_forecast(lat, lon)
                        else:
                            st.warning("Long-term forecast unavailable due to missing coordinates.")
                    else:
                        st.error("Please enter a city name.")

        with col2:
            if weather:
                st.header(f"Current Weather in {weather['Location']}")
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #f1f8e9 0%, #e8f5e9 100%);
                    border-left: 6px solid #4caf50;
                    padding: 20px;
                    border-radius: 12px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                    <h2 style="margin-top:0; margin-bottom:15px;">{weather['Temperature (°C)']} °C</h2>
                        <p><strong>Weather:</strong> {weather['Weather']}</p>
                        <p><strong>Humidity:</strong> {weather['Humidity (%)']}%</p>
                        <p><strong>Wind Speed:</strong> {weather['Wind Speed (m/s)']} m/s</p>
                    </div>
                """, unsafe_allow_html=True)

            if forecast:
                display_forecast(forecast)
            elif weather:
                st.info("No long-term forecast data available yet.")

    if __name__ == "__main__":
        main()
else:
    # Weather code and icon maps
    WEATHERCODE_DESCRIPTIONS = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
        55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
        67: "Heavy freezing rain", 71: "Slight snow fall", 73: "Moderate snow fall",
        75: "Heavy snow fall", 77: "Snow grains", 80: "Slight rain showers",
        81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
        86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }

    WEATHERCODE_ICONS = {
        0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️", 45: "🌫️", 48: "🌫️", 51: "🌦️",
        53: "🌧️", 55: "🌧️", 56: "🌧️❄️", 57: "🌧️❄️", 61: "🌧️", 63: "🌧️",
        65: "🌧️", 66: "🌧️❄️", 67: "🌧️❄️", 71: "❄️", 73: "❄️", 75: "❄️",
        77: "❄️", 80: "🌦️", 81: "🌧️", 82: "⛈️", 85: "🌨️", 86: "❄️", 95: "⛈️",
        96: "⛈️", 99: "⛈️"
    }

    OPENWEATHER_API_KEY = "80338847e882892b9d3fdc04654dd858"  # Your API key
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    # Initialize Groq LLM and memory
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)

    # Internal datasets
    disease_causes = {
        "late blight": "Caused by oomycete Phytophthora infestans, favored by wet and cool conditions.",
        "powdery mildew": "Caused by fungal pathogens, thrives in humid environments.",
    }

    fertilizer_recs = {
        "tomato": "Use balanced NPK 10-10-10 fertilizer.",
        "potato": "High potassium fertilizer recommended.",
    }

    def extract_city_name(text):
        # Try to find city name after weather-related keywords
        match = re.search(r'weather(?:\s+(?:in|at|for|of))?\s+([a-zA-Z\s]+)', text, re.IGNORECASE)
        if match:
            city = match.group(1).strip(" ?.,").lower()
        else:
            words = text.split()
            city = words[-1].strip(" ?.,").lower() if words else None

        if not city:
            return None

        # Remove common temporal words or extra noise
        stop_words = ["today", "tomorrow", "now", "currently", "right now", "please"]
        for word in stop_words:
            city = city.replace(word, "")

        return city.strip()

    def get_weather_by_city(city):
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                "Location": data["name"],
                "Temperature (°C)": data["main"]["temp"],
                "Humidity (%)": data["main"]["humidity"],
                "Weather": data["weather"][0]["description"].title(),
                "Wind Speed (m/s)": data["wind"]["speed"],
                "Lat": data["coord"]["lat"],
                "Lon": data["coord"]["lon"]
            }
        else:
            return None

    def get_long_term_forecast(lat, lon):
        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,weathercode",
            "timezone": "auto"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def display_forecast(forecast):
        if not forecast or "daily" not in forecast:
            st.error("No forecast data available.")
            return

        daily = forecast["daily"]
        df = pd.DataFrame({
            "Date": daily["time"],
            "Max Temp (°C)": daily["temperature_2m_max"],
            "Min Temp (°C)": daily["temperature_2m_min"],
            "Precipitation (mm)": daily["precipitation_sum"],
            "Rain (mm)": daily["rain_sum"],
            "Showers (mm)": daily["showers_sum"],
            "Snow (mm)": daily["snowfall_sum"],
            "Weather": [f"{WEATHERCODE_ICONS.get(code, '')} {WEATHERCODE_DESCRIPTIONS.get(code, 'Unknown')}" for code in daily["weathercode"]]
        })

        st.subheader("7-Days Weather Forecast")
        st.dataframe(df)

    def chatbot_response(user_input):
        lower_input = user_input.lower()

        # === WEATHER-RELATED KEYWORDS ===
        weather_keywords = {
            "weather", "temperature", "temp", "humidity", "wind", "rain", "forecast",
            "degree", "degrees", "climate", "precipitation", "snow", "sunny", "cloudy"
        }

        # === WEATHER HANDLING ===
        if any(keyword in lower_input for keyword in weather_keywords):
            city = extract_city_name(user_input)
            if not city:
                return "Please specify the city name for weather info."

            weather = get_weather_by_city(city)
            if not weather:
                return f"Sorry, I couldn't find weather data for '{city}'. Please check the city name."

            # CASE 1: Specific attributes
            if "humidity" in lower_input:
                return f"Humidity in {weather['Location']} is {weather['Humidity (%)']}%."
            if "temperature" in lower_input or "temp" in lower_input or "degree" in lower_input or "degrees" in lower_input:
                return f"The temperature in {weather['Location']} is {weather['Temperature (°C)']} °C."
            if "wind" in lower_input:
                return f"The wind speed in {weather['Location']} is {weather['Wind Speed (m/s)']} m/s."

            # CASE 2: Forecast
            if "forecast" in lower_input or "next" in lower_input or "7-day" in lower_input or "week" in lower_input:
                st.markdown(f"### 🌦️ 7-Day Weather Forecast for **{weather['Location']}**")
                forecast = get_long_term_forecast(weather["Lat"], weather["Lon"])
                if forecast:
                    display_forecast(forecast)
                    return f"Displayed 7-day weather forecast for {city}."
                else:
                    return "Forecast data not available for this location."

            # CASE 3: General current weather
            st.markdown(f"### 🌤️ Current Weather in **{weather['Location']}**")
            st.markdown(f"""
            - **Temperature:** {weather['Temperature (°C)']} °C  
            - **Humidity:** {weather['Humidity (%)']}%  
            - **Weather:** {weather['Weather']}  
            - **Wind Speed:** {weather['Wind Speed (m/s)']} m/s  
            """)
            return f"Displayed current weather for {city}."

        # === PLANT DISEASE ANSWERS ===
        for disease in disease_causes:
            if disease in lower_input:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #f1f8e9 0%, #e8f5e9 100%);
                    border-left: 6px solid #4caf50;
                    padding: 20px;
                    border-radius: 12px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                    <h3 style="color:#B22222;">🌿 Disease Information: {disease.title()}</h3>
                    <p style="font-size:16px; color:#333;">{disease_causes[disease]}</p>
                    </div>
                """, unsafe_allow_html=True)
                return f"Displayed details about {disease}."

        # === FERTILIZER RECOMMENDATIONS ===
        for crop in fertilizer_recs:
            if crop in lower_input:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #f1f8e9 0%, #e8f5e9 100%);
                    border-left: 6px solid #4caf50;
                    padding: 20px;
                    border-radius: 12px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                    <h3 style="color:#2E8B57;">🌾 Fertilizer Recommendation for {crop.title()}</h3>
                    <p style="font-size:16px; color:#333;">{fertilizer_recs[crop]}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                return f"Displayed fertilizer recommendation for {crop}."

        # === AGRICULTURE KEYWORDS ===
        agri_keywords = {
            # General agriculture terms
            "crop", "disease", "fertilizer", "plant", "soil", "pest", "seed",
            "harvest", "agriculture", "farming", "yield", "irrigation", "nutrient",

            # Common plant diseases
            "late blight", "powdery mildew", "rust", "leaf spot", "bacterial wilt",
            "root rot", "downy mildew", "anthracnose", "blight", "canker", "mosaic",

            # Common vegetables
            "tomato", "potato", "onion", "carrot", "cabbage", "chili", "spinach",
            "lettuce", "pepper", "eggplant", "brinjal", "beans", "okra", "cauliflower",

            # Common fruits
            "mango", "banana", "apple", "grape", "orange", "papaya", "guava",
            "pomegranate", "watermelon", "melon", "lemon", "lime", "pineapple"
        }

        # === SHORT, TO-THE-POINT AI RESPONSE ===
        if any(keyword in lower_input for keyword in agri_keywords):
            short_prompt = (
                "Answer briefly and to the point. Give only the most useful info. "
                "Limit your response to 4 sentences maximum. "
                f"Question: {user_input}"
            )
            concise_response = conversation.run(short_prompt)
            st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #f1f8e9 0%, #e8f5e9 100%);
                    border-left: 6px solid #4caf50;
                    padding: 20px;
                    border-radius: 12px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                    <p style="font-size:16px; color:#222; line-height:1.5;">{concise_response}</p>
                    </div>
                """, unsafe_allow_html=True)
            return "Displayed short agricultural advice."

        # === OUT-OF-SCOPE QUERIES ===
        st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, #f8e9e9 0%, #f5e8e8 100%);
                    border-left: 6px solid #FF0000;
                    padding: 20px;
                    border-radius: 12px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                    <h4 style="color:#333;">⚠️ Out of Scope</h4>
                    <p style="font-size:16px; color:#333;">
                        I'm designed to assist only with crop diseases, fertilizer guidance, and weather information related to farming.  
                        Please ask something related to agriculture.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        return "Out-of-scope query detected."

    def main():
        st.title("Smart Crop Care Chatbot")

        user_input = st.text_input("Ask about plant diseases, fertilizer, or weather:")

        if st.button("Send") and user_input:
            response = chatbot_response(user_input)
            if response:
                st.markdown(f"**Bot:** {response}")

    if __name__ == "__main__":
        main()