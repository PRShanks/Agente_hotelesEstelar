"""api/main.py.

--------------
Punto de entrada de la API REST del agente de Hoteles Estelar.

Expone dos endpoints:
  - POST /chat       -> conversacion directa con el agente (para pruebas)
  - POST /whatsapp   -> webhook de Twilio para mensajes de WhatsApp

Ejecutar localmente:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from api.whatsapp import procesar_mensaje_whatsapp
from llm.agent import AgentError, responder

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hoteles Estelar — Agente Conversacional",
    description="API REST del asistente corporativo de Hoteles Estelar S.A.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Payload para el endpoint /chat."""

    mensaje: str = Field(
        description="Pregunta o mensaje del usuario.",
        examples=["¿Qué servicios ofrece Hoteles Estelar?"],
    )
    session_id: str = Field(
        default="default",
        description="ID de sesion para mantener el historial.",
        examples=["usuario-123"],
    )


class ChatResponse(BaseModel):
    """Respuesta del endpoint /chat."""

    respuesta: str
    session_id: str
    confianza: str
    uso_tool_financiera: bool


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/")
def raiz() -> dict:
    """Health check."""
    return {"status": "ok", "servicio": "Hoteles Estelar Agente v3.0"}


@app.get("/health")
def health() -> dict:
    """Health check para Railway."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Endpoint de chat directo
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    """Endpoint de chat directo con el agente."""
    mensaje = body.mensaje.strip()
    session_id = body.session_id

    if not mensaje:
        return JSONResponse(
            status_code=400,
            content={"error": "El campo 'mensaje' no puede estar vacio."},
        )

    logger.info("Chat directo | session=%s | mensaje=%s", session_id, mensaje[:60])

    try:
        resultado = responder(pregunta=mensaje, session_id=session_id)
        return ChatResponse(
            respuesta=resultado["respuesta"],
            session_id=session_id,
            confianza=resultado.get("confianza", "media"),
            uso_tool_financiera=resultado.get("uso_tool_financiera", False),
        )
    except AgentError as e:
        logger.error("AgentError: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception:
        logger.exception("Error inesperado en /chat")
        return JSONResponse(
            status_code=500,
            content={"error": "Error interno del servidor."},
        )


# ---------------------------------------------------------------------------
# Webhook de Twilio para WhatsApp
# ---------------------------------------------------------------------------
@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...),
):
    numero = From.strip()
    mensaje = Body.strip()
    logger.info("WhatsApp | from=%s | msg=%s", numero, mensaje[:60])
    twiml = procesar_mensaje_whatsapp(numero=numero, mensaje=mensaje)
    from fastapi.responses import Response
    return Response(content=twiml, media_type="application/xml")
