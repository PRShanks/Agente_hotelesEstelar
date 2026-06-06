"""llm/agent.py.

--------------
Agente conversacional de Hoteles Estelar con Function Calling estricto,
dynamic_prompt, PostgresSaver y autenticacion HumanInTheLoop.

Arquitectura (Modulo 3):
  - init_chat_model: inicializa el LLM segun el proveedor configurado.
  - create_react_agent: orquesta el agente con herramientas (ReAct loop).
  - dynamic_prompt: ChatPromptTemplate que se construye en cada llamada
    con contexto del usuario, estado de autenticacion y fecha/hora.
  - Function Calling con esquemas Pydantic estrictos (no texto libre).
  - PostgresSaver: memoria persistente en Railway PostgreSQL.
  - HumanInTheLoop: pausa el flujo para autenticacion financiera.
  - Manejo de errores gracioso.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
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
    """Esquema estricto de entrada para la herramienta de busqueda RAG.

    El LLM DEBE proporcionar estos campos exactos — no puede improvisar
    parametros fuera del esquema. Esto es Function Calling estricto.
    """

    pregunta: str = Field(
        description=(
            "La pregunta o consulta del usuario en lenguaje natural. "
            "Debe ser especifica y clara para obtener los mejores resultados."
        )
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Numero de fragmentos a recuperar de Supabase (1-10).",
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
            f"En este momento no pude acceder a la base de datos corporativa. "
            f"Intentare responder con la informacion disponible. Error: {e}"
        )


# ---------------------------------------------------------------------------
# Herramientas disponibles
# ---------------------------------------------------------------------------
HERRAMIENTAS = [busqueda_rag, query_financiero]


# ---------------------------------------------------------------------------
# Dynamic Prompt — se construye en cada llamada con contexto dinamico
# ---------------------------------------------------------------------------
def construir_dynamic_prompt(
    nombre_empleado: str | None = None,
    autenticado: bool = False,
) -> ChatPromptTemplate:
    """Construye el prompt dinamico del agente segun el contexto del usuario.

    El prompt cambia en cada llamada segun:
    - Fecha y hora actual
    - Estado de autenticacion del usuario
    - Nombre del empleado autenticado (si aplica)

    Parametros:
        nombre_empleado: Nombre del empleado si esta autenticado.
        autenticado: True si el usuario paso la autenticacion financiera.

    Devuelve:
        ChatPromptTemplate con variables dinamicas listas para usar.
    """
    system_base = cargar_system_prompt()
    fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M")

    if autenticado and nombre_empleado:
        estado_auth = f"AUTENTICADO — {nombre_empleado}"
        contexto_empleado = (
            f"\nCONTEXTO DE SESION: El usuario autenticado es {nombre_empleado}. "
            f"Tiene acceso completo a informacion financiera confidencial. "
            f"Trata la informacion con la confidencialidad apropiada."
        )
        acceso_financiero = "HABILITADO"
    else:
        estado_auth = "NO AUTENTICADO"
        contexto_empleado = (
            "\nCONTEXTO DE SESION: Usuario no autenticado. "
            "Si solicita datos financieros, informale que requiere autenticacion."
        )
        acceso_financiero = "RESTRINGIDO (requiere codigo de empleado)"

    system_dinamico = f"""{system_base}

--- CONTEXTO DINAMICO DE SESION ---
Fecha y hora: {fecha_hora}
Estado de autenticacion: {estado_auth}
Acceso a datos financieros: {acceso_financiero}
{contexto_empleado}
--- FIN CONTEXTO DINAMICO ---"""

    return ChatPromptTemplate.from_messages([
        ("system", system_dinamico),
        ("placeholder", "{messages}"),
    ])


# ---------------------------------------------------------------------------
# Excepcion personalizada
# ---------------------------------------------------------------------------
class AgentError(Exception):
    """Error del agente conversacional."""


# ---------------------------------------------------------------------------
# Funcion principal del agente
# ---------------------------------------------------------------------------
def responder(pregunta: str, session_id: str = "default") -> dict:
    """Responde una pregunta con dynamic_prompt y autenticacion HumanInTheLoop.

    Flujo:
    1. Verificar si el bot espera un codigo de autenticacion (HumanInTheLoop).
    2. Verificar si la pregunta requiere autenticacion financiera.
    3. Construir el dynamic_prompt segun el contexto del usuario.
    4. Crear el agente con create_react_agent y el prompt dinamico.
    5. Invocar el agente con historial de memoria (PostgresSaver).

    Parametros:
        pregunta: Consulta del usuario.
        session_id: ID de sesion (numero de WhatsApp).

    Devuelve:
        Dict con respuesta, confianza, uso_tool_financiera, fuentes.
    """
    memoria = _get_memoria()

    # -----------------------------------------------------------------------
    # PASO 1: HumanInTheLoop — verificar si espera codigo de autenticacion
    # -----------------------------------------------------------------------
    esperando_auth = memoria.get_user_data(session_id, _ESPERANDO_AUTH_KEY)

    if esperando_auth and esperando_auth.get("esperando"):
        resultado_auth = verificar_codigo(pregunta, memoria, session_id)

        if resultado_auth["valido"]:
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
    # PASO 2: Verificar si requiere autenticacion financiera
    # -----------------------------------------------------------------------
    if es_pregunta_financiera(pregunta) and not esta_autenticado(memoria, session_id):
        memoria.save_user_data(
            session_id, _ESPERANDO_AUTH_KEY, {"esperando": True}
        )
        msg_auth = mensaje_solicitud_auth()
        memoria.save_message(session_id, "human", pregunta)
        memoria.save_message(session_id, "ai", msg_auth)
        logger.info("Auth requerida | session=%s", session_id)
        return {
            "respuesta": msg_auth,
            "confianza": "alta",
            "uso_tool_financiera": False,
            "tool_usada": None,
            "fuentes": [],
        }

    # -----------------------------------------------------------------------
    # PASO 3: Construir dynamic_prompt segun contexto del usuario
    # -----------------------------------------------------------------------
    autenticado = esta_autenticado(memoria, session_id)
    nombre_empleado = obtener_empleado_autenticado(memoria, session_id)

    prompt_dinamico = construir_dynamic_prompt(
        nombre_empleado=nombre_empleado,
        autenticado=autenticado,
    )

    logger.info(
        "Dynamic prompt | session=%s | auth=%s | empleado=%s",
        session_id, autenticado, nombre_empleado or "N/A",
    )

    # -----------------------------------------------------------------------
    # PASO 4: Crear agente con init_chat_model + create_react_agent
    # -----------------------------------------------------------------------
    modelo = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

    llm = init_chat_model(
        model=modelo,
        temperature=0.0,
        max_tokens=1024,
    )

    agente = create_react_agent(
        model=llm,
        tools=HERRAMIENTAS,
        prompt=prompt_dinamico,
    )

    # -----------------------------------------------------------------------
    # PASO 5: Invocar con historial de memoria
    # -----------------------------------------------------------------------
    historial = memoria.get_history(session_id)
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
        "Respuesta generada | session=%s | tool=%s | chars=%d | auth=%s",
        session_id, tool_usada, len(texto_respuesta), autenticado,
    )

    return {
        "respuesta": texto_respuesta,
        "confianza": "alta" if tool_usada else "media",
        "uso_tool_financiera": uso_tool_financiera,
        "tool_usada": tool_usada,
        "fuentes": [],
    }