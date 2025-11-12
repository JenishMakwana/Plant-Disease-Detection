import re
import streamlit as st
import requests
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

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
            <div style="background-color:#F7E6E6; padding:15px; border-radius:10px; box-shadow:0 0 10px #ddd;">
                <h3 style="color:#B22222;">🌿 Disease Information: {disease.title()}</h3>
                <p style="font-size:16px; color:#333;">{disease_causes[disease]}</p>
            </div>
            """, unsafe_allow_html=True)
            return f"Displayed details about {disease}."

    # === FERTILIZER RECOMMENDATIONS ===
    for crop in fertilizer_recs:
        if crop in lower_input:
            st.markdown(f"""
            <div style="background-color:#E7F6E7; padding:15px; border-radius:10px; box-shadow:0 0 10px #ddd;">
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
        <div style="background-color:#FFF9E6; padding:15px; border-radius:10px; box-shadow:0 0 10px #ddd;">
            <p style="font-size:16px; color:#222; line-height:1.5;">{concise_response}</p>
        </div>
        """, unsafe_allow_html=True)
        return "Displayed short agricultural advice."

    # === OUT-OF-SCOPE QUERIES ===
    st.markdown(f"""
    <div style="background-color:#F0F0F0; padding:15px; border-radius:10px; box-shadow:0 0 10px #ccc;">
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