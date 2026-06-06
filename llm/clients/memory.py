"""llm/clients/memory.py.

-----------------------
Gestion de memoria del agente con PostgresSaver de LangGraph.

Modulo 3: Migracion de SqliteStore a PostgresSaver para memoria
persistente en la nube usando la base de datos PostgreSQL de Railway.

Si DATABASE_URL no esta configurada, cae a SqliteStore como fallback.
"""

from __future__ import annotations

import logging
import os
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

SESSIONS_NS = "sessions"
MESSAGES_NS = "messages"
USERS_NS = "users"
PROFILE_NS = "profile"
_AUTH_KEY = "auth_financiera"


def _crear_store():
    """Crea el store de memoria.

    Intenta usar PostgresSaver (Railway PostgreSQL).
    Si falla o no hay DATABASE_URL, usa SqliteStore como fallback.

    Devuelve:
        Tupla (store, tipo) donde tipo es "postgres" o "sqlite".
    """
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        try:
            from psycopg import Connection
            from langgraph.checkpoint.postgres import PostgresSaver

            # autocommit=True necesario para CREATE INDEX CONCURRENTLY
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
    import sqlite3
    from langgraph.store.sqlite import SqliteStore

    path = _DEFAULT_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    store = SqliteStore(conn)
    store.setup()
    logger.info("SqliteStore inicializado como fallback")
    return store, "sqlite"


class SessionMemory:
    """Gestiona la memoria de conversacion con PostgresSaver o SqliteStore.

    PostgresSaver (LangGraph) para memoria persistente en Railway PostgreSQL.
    SqliteStore como fallback si DATABASE_URL no esta configurada.
    """

    def __init__(self) -> None:
        """Inicializa el store segun la configuracion disponible."""
        self._store, self._tipo = _crear_store()
        logger.info("SessionMemory inicializada con %s", self._tipo)

    @property
    def tipo(self) -> str:
        """Tipo de store: 'postgres' o 'sqlite'."""
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

    def save_user_data(self, user_id: str, key: str, value: dict) -> None:
        """Guarda datos del usuario (autenticacion, preferencias)."""
        if self._tipo == "sqlite":
            namespace = (USERS_NS, user_id, PROFILE_NS)
            self._store.put(namespace, key, value)

    def get_user_data(self, user_id: str, key: str) -> dict | None:
        """Recupera datos del usuario."""
        if self._tipo == "sqlite":
            namespace = (USERS_NS, user_id, PROFILE_NS)
            item = self._store.get(namespace, key)
            return item.value if item else None
        return None

    def delete_user_data(self, user_id: str, key: str) -> None:
        """Elimina datos del usuario."""
        if self._tipo == "sqlite":
            namespace = (USERS_NS, user_id, PROFILE_NS)
            self._store.delete(namespace, key)