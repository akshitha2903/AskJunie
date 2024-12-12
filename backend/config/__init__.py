import os
from typing import Dict, Any

def load_secrets() -> Dict[str, str]:

    # Load API keys
    try:
        with open("config/apikey.txt", "r") as file:
            openai_key = file.read().strip()
        with open("config/gmapi.txt", "r") as file:
            gmaps_key = file.read().strip()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {e}")

    # Initialize database
    db_config = {
        "host": "localhost",
        "database": "travel_agent_db",
        "user": "postgres",
        "password": "akshitha2903",
        "port": "5432"
    }

    return {
        "OPENAI_API_KEY": openai_key,
        "GOOGLEMAPS_API_KEY": gmaps_key,
        "DB_CONFIG": db_config
    }

import mysql.connector
def connect_to_db():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="akshitha2903",
        database="mydatabase"
    )
    return connection
