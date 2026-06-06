"""llm/agent.py.

--------------
Agente conversacional de Hoteles Estelar con Function Calling estricto
y autenticacion para datos financieros sensibles.

Arquitectura (Modulo 3):
  - init_chat_model: inicializa el LLM segun el proveedor configurado.
  - create_react_agent: orquesta el agente con herramientas (ReAct loop).
  - Function Calling con esquemas Pydantic estrictos (no texto libre).
  - Memoria persistente por session_id usando SqliteStore de LangGraph.
  - Autenticacion con codigo de 4 digitos para datos financieros.
  - HumanInTheLoop: pausa el flujo para pedir autenticacion al usuario.
  - Manejo de errores gracioso.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from llm.auth import (
    cerrar_sesion,
    es_pregunta_financiera,
    esta_autenticado,
    mensaje_solicitud_auth,
    obtener_empleado_autenticado,
    verificar_codigo,
)
from llm.clients.memory import SessionMemory
from llm.financial.tool import query_financiero
from llm.prompts.qa import cargar_system_prompt
from llm.rag.embeddings import crear_embeddings
from llm.rag.vector_store import crear_vector_store

load_dotenv()
logger = logging.getLogger(__name__)

# Clave en memoria para saber si el bot espera un codigo de autenticacion
_ESPERANDO_AUTH_KEY = "esperando_auth"

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
    """Esquema estricto para la herramienta de busqueda RAG."""

    pregunta: str = Field(
        description=(
            "La pregunta o consulta del usuario en lenguaje natural. "
            "Debe ser especifica y clara."
        )
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Numero de fragmentos a recuperar (1-10).",
    )


@tool(args_schema=BusquedaRAGInput)
def busqueda_rag(pregunta: str, top_k: int = 5) -> str:
    """Busca informacion corporativa de Hoteles Estelar en la base de datos vectorial.

    Usala para preguntas sobre:
    - Servicios, habitaciones, restaurantes, instalaciones.
    - Politica de mascotas, menores, fumadores.
    - Informacion de hoteles en ciudades especificas.
    - Programa de fidelizacion Huesped Siempre Estelar.
    - Alianzas, mision, vision, sostenibilidad.
    - Contacto, reservas, tarifas generales.

    NO usar para cifras financieras exactas (ingresos, EBITDA, deuda).
    """
    try:
        embeddings = crear_embeddings()
        vector_store = crear_vector_store(embeddings)

        if vector_store is None:
            return "La base de datos vectorial no esta configurada."

        docs = vector_store.similarity_search(pregunta, k=top_k)

        if not docs:
            return "No se encontro informacion relevante para esta consulta."

        fragmentos = []
        for i, doc in enumerate(docs, 1):
            fuente = doc.metadata.get("source", "desconocida")
            fragmentos.append(f"[{i}] (Fuente: {fuente})\n{doc.page_content}")

        return "\n\n---\n\n".join(fragmentos)

    except Exception as e:
        logger.exception("Error en busqueda_rag")
        return (
            f"En este momento no pude acceder a la base de datos. "
            f"Error: {e}"
        )


# ---------------------------------------------------------------------------
# Herramientas disponibles para el agente
# ---------------------------------------------------------------------------
HERRAMIENTAS = [busqueda_rag, query_financiero]


# ---------------------------------------------------------------------------
# Excepcion personalizada
# ---------------------------------------------------------------------------
class AgentError(Exception):
    """Error del agente conversacional."""


# ---------------------------------------------------------------------------
# Funcion principal del agente
# ---------------------------------------------------------------------------
def responder(pregunta: str, session_id: str = "default") -> dict:
    """Responde una pregunta con autenticacion para datos financieros.

    Flujo HumanInTheLoop:
    1. Si la pregunta es financiera y NO esta autenticado:
       → pide el codigo de 4 digitos (pausa el flujo)
    2. Si el bot estaba esperando un codigo:
       → verifica el codigo enviado
    3. Si esta autenticado o la pregunta es general:
       → usa create_react_agent normalmente

    Parametros:
        pregunta: Consulta del usuario.
        session_id: ID de sesion (numero de WhatsApp).

    Devuelve:
        Dict con respuesta, confianza, uso_tool_financiera, fuentes.
    """
    memoria = _get_memoria()

    # -----------------------------------------------------------------------
    # PASO 1: Verificar si el bot estaba esperando un codigo de autenticacion
    # -----------------------------------------------------------------------
    esperando_auth = memoria.get_user_data(session_id, _ESPERANDO_AUTH_KEY)

    if esperando_auth and esperando_auth.get("esperando"):
        # El usuario envio su codigo — verificar
        resultado_auth = verificar_codigo(pregunta, memoria, session_id)

        if resultado_auth["valido"]:
            # Autenticado — limpiar el flag de espera
            memoria.save_user_data(
                session_id, _ESPERANDO_AUTH_KEY, {"esperando": False}
            )
            memoria.save_message(session_id, "human", pregunta)
            memoria.save_message(session_id, "ai", resultado_auth["mensaje"])
            return {
                "respuesta": (
                    resultado_auth["mensaje"] + "\n\n"
                    "Ahora puedes hacer tu pregunta sobre los datos financieros."
                ),
                "confianza": "alta",
                "uso_tool_financiera": False,
                "tool_usada": None,
                "fuentes": [],
            }
        else:
            # Codigo incorrecto
            memoria.save_message(session_id, "human", pregunta)
            memoria.save_message(session_id, "ai", resultado_auth["mensaje"])
            return {
                "respuesta": resultado_auth["mensaje"],
                "confianza": "alta",
                "uso_tool_financiera": False,
                "tool_usada": None,
                "fuentes": [],
            }

    # -----------------------------------------------------------------------
    # PASO 2: Verificar si la pregunta requiere autenticacion financiera
    # -----------------------------------------------------------------------
    if es_pregunta_financiera(pregunta) and not esta_autenticado(memoria, session_id):
        # HumanInTheLoop: pausar y pedir codigo
        memoria.save_user_data(
            session_id, _ESPERANDO_AUTH_KEY, {"esperando": True}
        )
        msg_auth = mensaje_solicitud_auth()
        memoria.save_message(session_id, "human", pregunta)
        memoria.save_message(session_id, "ai", msg_auth)
        logger.info(
            "Auth requerida | session=%s | pregunta=%s",
            session_id, pregunta[:50],
        )
        return {
            "respuesta": msg_auth,
            "confianza": "alta",
            "uso_tool_financiera": False,
            "tool_usada": None,
            "fuentes": [],
        }

    # -----------------------------------------------------------------------
    # PASO 3: Procesamiento normal con create_react_agent
    # -----------------------------------------------------------------------
    historial = memoria.get_history(session_id)

    modelo = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

    llm = init_chat_model(
        model=modelo,
        temperature=0.0,
        max_tokens=1024,
    )

    system_prompt = cargar_system_prompt()

    # Si esta autenticado, agregar contexto del empleado al system prompt
    nombre_empleado = obtener_empleado_autenticado(memoria, session_id)
    if nombre_empleado:
        system_prompt = (
            f"{system_prompt}\n\n"
            f"CONTEXTO: El usuario autenticado es {nombre_empleado}. "
            f"Tiene acceso a informacion financiera confidencial."
        )

    agente = create_react_agent(
        model=llm,
        tools=HERRAMIENTAS,
        prompt=system_prompt,
    )

    mensajes = list(historial) + [HumanMessage(content=pregunta)]

    uso_tool_financiera = False
    tool_usada = None

    try:
        resultado = agente.invoke({"messages": mensajes})
        mensajes_resultado = resultado.get("messages", [])
        texto_respuesta = ""

        for msg in reversed(mensajes_resultado):
            tool_calls = getattr(msg, "tool_calls", None)
            tiene_tool_calls = bool(tool_calls)
            msg_type = getattr(msg, "type", "")
            if hasattr(msg, "content") and msg.content and not tiene_tool_calls and msg_type == "ai":
                texto_respuesta = msg.content
                break

        if not texto_respuesta:
            texto_respuesta = "No pude generar una respuesta en este momento."

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
