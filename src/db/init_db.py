"""Apply schema.sql to create all tables. Safe to run repeatedly."""
from pathlib import Path

from src.db.connection import get_connection

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db() -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
    print("Schema applied. All tables are ready.")


if __name__ == "__main__":
    init_db()
