"""llm/clients/memory.py.

-----------------------
Gestion de memoria del agente con PostgresSaver de LangGraph.

Modulo 3: Migracion de SqliteStore a PostgresSaver para memoria
persistente en la nube usando la base de datos PostgreSQL de Railway.

Si DATABASE_URL no esta configurada, cae a SqliteStore como fallback.

NOTA: Los datos de usuario (autenticacion, preferencias) siempre se
guardan en SQLite local, independientemente del store principal.
Esto garantiza que la autenticacion funciona con PostgresSaver.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DEFAULT_HISTORY_LIMIT = 20
_DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "memoria.db"
)
_USER_DATA_DB_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "user_data.db"
)

SESSIONS_NS = "sessions"
MESSAGES_NS = "messages"
USERS_NS = "users"
PROFILE_NS = "profile"


def _crear_store():
    """Crea el store de memoria principal.

    Intenta usar PostgresSaver (Railway PostgreSQL).
    Si falla o no hay DATABASE_URL, usa SqliteStore como fallback.
    """
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        try:
            from psycopg import Connection
            from langgraph.checkpoint.postgres import PostgresSaver

            conn = Connection.connect(database_url, autocommit=True)
            saver = PostgresSaver(conn)
            saver.setup()
            logger.info("PostgresSaver inicializado correctamente")
            return saver, "postgres"
        except Exception as e:
            logger.warning(
                "No se pudo inicializar PostgresSaver: %s. "
                "Usando SqliteStore como fallback.", e
            )

    # Fallback: SqliteStore
    from langgraph.store.sqlite import SqliteStore

    path = _DEFAULT_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    store = SqliteStore(conn)
    store.setup()
    logger.info("SqliteStore inicializado como fallback")
    return store, "sqlite"


def _crear_user_data_db() -> sqlite3.Connection:
    """Crea la base de datos SQLite para datos de usuario.

    Siempre usa SQLite local para datos de autenticacion y preferencias,
    independientemente del store principal (PostgresSaver o SqliteStore).
    """
    Path(_USER_DATA_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_USER_DATA_DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        )
    """)
    conn.commit()
    return conn


class SessionMemory:
    """Gestiona la memoria de conversacion con PostgresSaver o SqliteStore.

    Store principal: PostgresSaver (Railway PostgreSQL) o SqliteStore.
    Datos de usuario: SQLite local siempre (autenticacion, preferencias).
    """

    def __init__(self) -> None:
        """Inicializa el store segun la configuracion disponible."""
        self._store, self._tipo = _crear_store()
        self._user_db = _crear_user_data_db()
        logger.info("SessionMemory inicializada con %s", self._tipo)

    @property
    def tipo(self) -> str:
        """Tipo de store principal: 'postgres' o 'sqlite'."""
        return self._tipo

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Guarda un mensaje en el historial de la sesion."""
        if self._tipo == "sqlite":
            namespace = (SESSIONS_NS, session_id, MESSAGES_NS)
            key = f"{role}-{uuid.uuid4().hex[:8]}"
            self._store.put(
                namespace, key,
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

    def get_history(
        self,
        session_id: str,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> list[BaseMessage]:
        """Recupera el historial reciente de la sesion."""
        if self._tipo == "sqlite":
            namespace = (SESSIONS_NS, session_id, MESSAGES_NS)
            items = self._store.search(namespace, limit=200)
            sorted_items = sorted(
                items,
                key=lambda item: item.value.get("timestamp", ""),
            )
            recent = sorted_items[-limit:] if len(sorted_items) > limit else sorted_items
            messages: list[BaseMessage] = []
            for item in recent:
                role = item.value.get("role", "human")
                content = item.value.get("content", "")
                if role == "ai":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
            return messages
        return []

    def clear_session(self, session_id: str) -> None:
        """Elimina el historial de una sesion."""
        if self._tipo == "sqlite":
            namespace = (SESSIONS_NS, session_id, MESSAGES_NS)
            items = self._store.search(namespace, limit=1000)
            for item in items:
                self._store.delete(namespace, item.key)

    def session_exists(self, session_id: str) -> bool:
        """Verifica si una sesion tiene historial."""
        if self._tipo == "sqlite":
            namespace = (SESSIONS_NS, session_id, MESSAGES_NS)
            items = self._store.search(namespace, limit=1)
            return len(items) > 0
        return False

    # -----------------------------------------------------------------------
    # Datos de usuario — siempre SQLite local (funciona con PostgresSaver)
    # -----------------------------------------------------------------------

    def save_user_data(self, user_id: str, key: str, value: dict) -> None:
        """Guarda datos del usuario en SQLite local.

        Funciona independientemente del store principal.
        Usado para autenticacion financiera y preferencias.
        """
        self._user_db.execute(
            "INSERT OR REPLACE INTO user_data (user_id, key, value) VALUES (?, ?, ?)",
            (user_id, key, json.dumps(value)),
        )
        self._user_db.commit()

    def get_user_data(self, user_id: str, key: str) -> dict | None:
        """Recupera datos del usuario desde SQLite local."""
        cursor = self._user_db.execute(
            "SELECT value FROM user_data WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def delete_user_data(self, user_id: str, key: str) -> None:
        """Elimina datos del usuario desde SQLite local."""
        self._user_db.execute(
            "DELETE FROM user_data WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        self._user_db.commit()