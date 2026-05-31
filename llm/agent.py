"""llm/agent.py.

--------------
Agente conversacional de Hoteles Estelar con Function Calling estricto.

Arquitectura (Módulo 3):
  - Function Calling con esquemas Pydantic estrictos (no texto libre).
  - Memoria persistente por session_id usando SqliteStore de LangGraph.
  - Dos herramientas: RAG (Supabase) y consulta financiera (SQLite).
  - Manejo de errores gracioso: si una tool falla, responde sin ella.

Diferencia con Módulo 2:
  - Módulo 2: el LLM decidía qué herramienta usar con texto libre.
  - Módulo 3: el LLM DEBE invocar una herramienta con un JSON Schema
    estricto validado por Pydantic — más fiable y predecible.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm.clients.factory import crear_llm
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
    parámetros fuera del esquema.
    """

    pregunta: str = Field(
        description=(
            "La pregunta o consulta del usuario en lenguaje natural. "
            "Debe ser específica y clara para obtener los mejores resultados "
            "de la búsqueda semántica."
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
        return f"Error al buscar en la base de datos: {e}"


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
    """Responde una pregunta usando el agente con Function Calling estricto.

    Flujo:
    1. Cargar historial de la sesión (memoria persistente).
    2. Invocar el LLM con las herramientas disponibles (Function Calling).
    3. Si el LLM invoca una herramienta, ejecutarla y hacer segundo invoke.
    4. Guardar la pregunta y respuesta en memoria.
    5. Devolver la respuesta estructurada.

    Parámetros:
        pregunta: Consulta del usuario en lenguaje natural.
        session_id: Identificador de sesión (número de WhatsApp o ID custom).

    Devuelve:
        Dict con claves: respuesta, confianza, uso_tool_financiera, fuentes.

    Lanza:
        AgentError: Si el agente no puede generar una respuesta.
    """
    memoria = _get_memoria()
    historial = memoria.get_history(session_id)

    # Crear el LLM con las herramientas enlazadas (Function Calling)
    llm = crear_llm(temperature=0.0, max_tokens=1024)
    llm_con_tools = llm.bind_tools(HERRAMIENTAS)

    system_prompt = cargar_system_prompt()

    # Construir los mensajes
    mensajes: list = [("system", system_prompt)]
    mensajes.extend(historial)
    mensajes.append(("human", pregunta))

    uso_tool_financiera = False
    tool_usada = None

    try:
        # --- Primer invoke: el LLM decide si usar herramienta ---
        respuesta_inicial = llm_con_tools.invoke(mensajes)

        if respuesta_inicial.tool_calls:
            tool_call = respuesta_inicial.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logger.info(
                "Tool invocada | session=%s | tool=%s | args=%s",
                session_id, tool_name, tool_args,
            )

            # Ejecutar la herramienta correcta
            try:
                if tool_name == "busqueda_rag":
                    resultado_tool = busqueda_rag.invoke(tool_args)
                    tool_usada = "RAG"
                elif tool_name == "query_financiero":
                    resultado_tool = query_financiero.invoke(tool_args)
                    tool_usada = "financiera"
                    uso_tool_financiera = True
                else:
                    resultado_tool = f"Herramienta '{tool_name}' no reconocida."
                    tool_usada = "desconocida"
            except Exception as e:
                logger.warning("Tool %s falló: %s", tool_name, e)
                resultado_tool = (
                    f"En este momento no pude obtener la información solicitada "
                    f"({tool_name} falló). Intentaré responder con lo que sé."
                )

            # --- Segundo invoke: el LLM genera la respuesta final con el resultado ---
            mensajes_con_tool = [
                *mensajes,
                respuesta_inicial,
                ToolMessage(
                    content=resultado_tool,
                    tool_call_id=tool_call["id"],
                ),
            ]

            respuesta_final = llm.invoke(mensajes_con_tool)
            texto_respuesta = respuesta_final.content

        else:
            # El LLM respondió directamente sin usar herramienta
            texto_respuesta = respuesta_inicial.content
            tool_usada = None

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
