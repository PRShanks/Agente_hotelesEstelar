"""llm/agent.py.

--------------
Agente conversacional de Hoteles Estelar con Function Calling estricto.

Arquitectura (Módulo 3):
  - init_chat_model: inicializa el LLM según el proveedor configurado.
  - create_react_agent: orquesta el agente con herramientas (ReAct loop).
  - Function Calling con esquemas Pydantic estrictos (no texto libre).
  - Memoria persistente por session_id usando SqliteStore de LangGraph.
  - Dos herramientas: RAG (Supabase) y consulta financiera (SQLite).
  - Manejo de errores gracioso: si una tool falla, responde sin ella.

Diferencia con Módulo 2:
  - Módulo 2: el LLM decidía qué herramienta usar con texto libre (bind_tools manual).
  - Módulo 3: create_react_agent orquesta el loop completo de forma estandarizada.
    El LLM DEBE invocar una herramienta con un JSON Schema estricto validado
    por Pydantic, aumentando drásticamente la fiabilidad.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from llm.clients.memory import SessionMemory
from llm.financial.tool import query_financiero
from llm.prompts.qa import cargar_system_prompt
from llm.rag.embeddings import crear_embeddings
from llm.rag.vector_store import crear_vector_store

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memoria compartida (singleton por proceso)
# ---------------------------------------------------------------------------
_memoria: SessionMemory | None = None


def _get_memoria() -> SessionMemory:
    """Devuelve la instancia singleton de SessionMemory."""
    global _memoria
    if _memoria is None:
        _memoria = SessionMemory()
    return _memoria


# ---------------------------------------------------------------------------
# Esquema Pydantic estricto para la herramienta RAG
# ---------------------------------------------------------------------------
class BusquedaRAGInput(BaseModel):
    """Esquema estricto de entrada para la herramienta de búsqueda RAG.

    El LLM DEBE proporcionar estos campos exactos — no puede improvisar
    parámetros fuera del esquema. Esto es Function Calling estricto con Pydantic.
    """

    pregunta: str = Field(
        description=(
            "La pregunta o consulta del usuario en lenguaje natural. "
            "Debe ser específica y clara para obtener los mejores resultados "
            "de la búsqueda semántica en Supabase."
        )
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Número de fragmentos a recuperar de la base de datos vectorial (1-10).",
    )


@tool(args_schema=BusquedaRAGInput)
def busqueda_rag(pregunta: str, top_k: int = 5) -> str:
    """Busca información corporativa de Hoteles Estelar en la base de datos vectorial.

    Úsala para preguntas sobre:
    - Servicios, habitaciones, restaurantes, instalaciones.
    - Política de mascotas, menores, fumadores.
    - Información de hoteles en ciudades específicas.
    - Programa de fidelización Huésped Siempre Estelar.
    - Alianzas, misión, visión, sostenibilidad.
    - Contacto, reservas, tarifas generales.

    NO usar para cifras financieras exactas (ingresos, EBITDA, deuda) —
    para eso usar query_financiero.
    """
    try:
        embeddings = crear_embeddings()
        vector_store = crear_vector_store(embeddings)

        if vector_store is None:
            return (
                "La base de datos vectorial no está configurada. "
                "Verifica SUPABASE_URL y SUPABASE_SERVICE_KEY en el .env."
            )

        docs = vector_store.similarity_search(pregunta, k=top_k)

        if not docs:
            return "No se encontró información relevante para esta consulta."

        fragmentos = []
        for i, doc in enumerate(docs, 1):
            fuente = doc.metadata.get("source", "desconocida")
            fragmentos.append(f"[{i}] (Fuente: {fuente})\n{doc.page_content}")

        return "\n\n---\n\n".join(fragmentos)

    except Exception as e:
        logger.exception("Error en busqueda_rag")
        return (
            f"En este momento no pude acceder a la base de datos corporativa. "
            f"Intentaré responder con la información disponible. Error: {e}"
        )


# ---------------------------------------------------------------------------
# Herramientas disponibles para el agente
# ---------------------------------------------------------------------------
HERRAMIENTAS = [busqueda_rag, query_financiero]


# ---------------------------------------------------------------------------
# Excepción personalizada
# ---------------------------------------------------------------------------
class AgentError(Exception):
    """Error del agente conversacional."""


# ---------------------------------------------------------------------------
# Función principal del agente
# ---------------------------------------------------------------------------
def responder(pregunta: str, session_id: str = "default") -> dict:
    """Responde una pregunta usando create_react_agent con Function Calling estricto.

    Flujo:
    1. Inicializar el LLM con init_chat_model (estandar LangChain).
    2. Crear el agente con create_react_agent (ReAct loop automatico).
    3. Cargar historial de la sesion (memoria persistente).
    4. Invocar el agente — el decide cuantas tools usar y en que orden.
    5. Guardar la pregunta y respuesta en memoria.
    6. Devolver la respuesta estructurada.

    Parametros:
        pregunta: Consulta del usuario en lenguaje natural.
        session_id: Identificador de sesion (numero de WhatsApp o ID custom).

    Devuelve:
        Dict con claves: respuesta, confianza, uso_tool_financiera, fuentes.

    Lanza:
        AgentError: Si el agente no puede generar una respuesta.
    """
    memoria = _get_memoria()
    historial = memoria.get_history(session_id)

    # --- init_chat_model: inicializa el LLM de forma estandarizada ---
    # Lee el proveedor del entorno: "anthropic" por defecto en produccion
    proveedor = os.getenv("LLM_PROVIDER", "anthropic")
    modelo = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

    llm = init_chat_model(
        model=modelo,
        model_provider=proveedor,
        temperature=0.0,
        max_tokens=1024,
    )

    # --- create_react_agent: crea el agente con el loop ReAct automatico ---
    # ReAct = Reasoning + Acting: el agente razona, actua con tools, y repite
    # hasta tener suficiente informacion para responder.
    system_prompt = cargar_system_prompt()

    agente = create_react_agent(
        model=llm,
        tools=HERRAMIENTAS,
        prompt=system_prompt,
    )

    # Construir los mensajes con historial
    mensajes = list(historial) + [HumanMessage(content=pregunta)]

    uso_tool_financiera = False
    tool_usada = None

    try:
        # Invocar el agente — create_react_agent maneja el loop completo
        resultado = agente.invoke({"messages": mensajes})

        # Extraer la respuesta final (ultimo mensaje del agente)
        mensajes_resultado = resultado.get("messages", [])
        texto_respuesta = ""

        for msg in reversed(mensajes_resultado):
            # Tomar el ultimo AIMessage con contenido y sin tool_calls activos
            tool_calls = getattr(msg, "tool_calls", None)
            tiene_tool_calls = bool(tool_calls)
            msg_type = getattr(msg, "type", "")
            if hasattr(msg, "content") and msg.content and not tiene_tool_calls and msg_type == "ai":
                texto_respuesta = msg.content
                break

        if not texto_respuesta:
            texto_respuesta = "No pude generar una respuesta en este momento."

        # Detectar si se uso la tool financiera
        for msg in mensajes_resultado:
            if hasattr(msg, "name"):
                if msg.name == "query_financiero":
                    uso_tool_financiera = True
                    tool_usada = "financiera"
                elif msg.name == "busqueda_rag":
                    tool_usada = "RAG"

    except Exception as e:
        logger.exception("Error en el agente | session=%s", session_id)
        raise AgentError(f"Error generando respuesta: {e}") from e

    # Guardar en memoria
    memoria.save_message(session_id, "human", pregunta)
    memoria.save_message(session_id, "ai", texto_respuesta)

    logger.info(
        "Respuesta generada | session=%s | tool=%s | chars=%d",
        session_id, tool_usada, len(texto_respuesta),
    )

    return {
        "respuesta": texto_respuesta,
        "confianza": "alta" if tool_usada else "media",
        "uso_tool_financiera": uso_tool_financiera,
        "tool_usada": tool_usada,
        "fuentes": [],
    }
