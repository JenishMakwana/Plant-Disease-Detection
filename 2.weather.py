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
    st.set_page_config(
        page_title="Weather Forecast App",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🌤️ Realtime Weather Forecast App")
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
                st.success(f"Detected location: {city} ({lat:.4f}, {lon:.4f})")
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
            st.markdown(
                f"""
                <div style="
                    background-color:#e0e7ff;
                    padding: 25px; 
                    border-radius: 15px;
                    box-shadow: 0 4px 8px rgba(48, 87, 213, 0.2);
                    color: #1e1e2f;
                    font-size: 20px;
                    font-weight: 600;
                    line-height: 1.5;
                ">
                    <h1 style="margin-top:0; margin-bottom:15px;">{weather['Temperature (°C)']} °C</h1>
                    <p><strong>Weather:</strong> {weather['Weather']}</p>
                    <p><strong>Humidity:</strong> {weather['Humidity (%)']}%</p>
                    <p><strong>Wind Speed:</strong> {weather['Wind Speed (m/s)']} m/s</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if forecast:
            display_forecast(forecast)
        elif weather:
            st.info("No long-term forecast data available yet.")

if __name__ == "__main__":
    main()