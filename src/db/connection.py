"""Database connection helpers."""
import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_dsn() -> dict:
    """Read connection parameters from the environment (.env)."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "fpl"),
        "user": os.getenv("DB_USER", "fpl"),
        "password": os.getenv("DB_PASSWORD", "fpl"),
    }


@contextmanager
def get_connection():
    """Yield a connection; commit on success, roll back on error, always close."""
    conn = psycopg2.connect(**get_dsn())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
