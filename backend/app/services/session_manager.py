import json
from datetime import datetime
from typing import Optional, List, Dict
import psycopg2
from psycopg2.extras import Json
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_config: dict):
        """
        Initialize database connection with configuration.
        
        db_config should contain:
        - host
        - database
        - user
        - password
        - port
        """
        self.db_config = db_config

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(**self.db_config)
        try:
            yield conn
        finally:
            conn.close()

    def initialize_database(self):
        """Create necessary tables if they don't exist"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Create sessions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        trip_details JSONB DEFAULT '{}'::jsonb
                    )
                """)

                # Create conversation_history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) REFERENCES sessions(session_id),
                        role VARCHAR(50),
                        content TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create current_itinerary table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS current_itinerary (
                        session_id VARCHAR(255) PRIMARY KEY REFERENCES sessions(session_id),
                        itinerary_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create modification_history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS modification_history (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) REFERENCES sessions(session_id),
                        modifications JSONB,
                        result JSONB,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()

    def create_or_update_session(self, session_id: str) -> None:
        """Create a new session or update last_active timestamp"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sessions (session_id)
                    VALUES (%s)
                    ON CONFLICT (session_id)
                    DO UPDATE SET last_active = CURRENT_TIMESTAMP
                """, (session_id,))
                conn.commit()

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history for a session"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT role, content, timestamp
                    FROM conversation_history
                    WHERE session_id = %s
                    ORDER BY timestamp ASC
                """, (session_id,))
                rows = cur.fetchall()
                return [
                    {"role": row[0], "content": row[1], "timestamp": row[2]}
                    for row in rows
                ]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a new message to the conversation history"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_history (session_id, role, content)
                    VALUES (%s, %s, %s)
                """, (session_id, role, content))
                conn.commit()

    def get_trip_details(self, session_id: str) -> dict:
        """Get trip details for a session"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT trip_details
                    FROM sessions
                    WHERE session_id = %s
                """, (session_id,))
                result = cur.fetchone()
                return result[0] if result else {}

    def update_trip_details(self, session_id: str, details: dict) -> None:
        """Update trip details for a session"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE sessions
                    SET trip_details = trip_details || %s::jsonb
                    WHERE session_id = %s
                """, (Json(details), session_id))
                conn.commit()

    def save_itinerary(self, session_id: str, itinerary_data: dict) -> None:
        """Save or update current itinerary"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO current_itinerary (session_id, itinerary_data)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id)
                    DO UPDATE SET 
                        itinerary_data = %s,
                        last_modified = CURRENT_TIMESTAMP
                """, (session_id, Json(itinerary_data), Json(itinerary_data)))
                conn.commit()

    def get_current_itinerary(self, session_id: str) -> Optional[dict]:
        """Retrieve current itinerary for a session"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT itinerary_data
                    FROM current_itinerary
                    WHERE session_id = %s
                """, (session_id,))
                result = cur.fetchone()
                return result[0] if result else None

    def add_modification_history(self, session_id: str, modifications: dict, result: dict) -> None:
        """Add a modification history entry"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO modification_history 
                    (session_id, modifications, result)
                    VALUES (%s, %s, %s)
                """, (session_id, Json(modifications), Json(result)))
                conn.commit()

    def cleanup_old_sessions(self, days: int = 30) -> None:
        """Clean up sessions older than specified days"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM sessions
                    WHERE last_active < CURRENT_TIMESTAMP - INTERVAL '%s days'
                """, (days,))
                conn.commit()