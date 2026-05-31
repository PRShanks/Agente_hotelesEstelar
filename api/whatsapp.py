"""api/whatsapp.py.

-----------------
Procesa mensajes entrantes de WhatsApp vía Twilio y genera respuestas TwiML.

Responsabilidades:
  1. Recibir el número y mensaje del usuario.
  2. Llamar al agente para obtener la respuesta.
  3. Formatear la respuesta en TwiML para que Twilio la envíe.
  4. Manejar errores graciosamente — el usuario siempre recibe una respuesta.
"""

from __future__ import annotations

import logging

from llm.agent import AgentError, responder

logger = logging.getLogger(__name__)

_MENSAJE_ERROR = (
    "Lo siento, en este momento no puedo procesar tu solicitud. "
    "Por favor intenta de nuevo en unos instantes o contáctanos directamente "
    "en nuestra sede principal. 🏨"
)


def _twiml(texto: str) -> str:
    """Envuelve un texto en formato TwiML para Twilio.

    Parámetros:
        texto: Respuesta del agente en texto plano.

    Devuelve:
        String XML con formato TwiML.
    """
    # Escapar caracteres especiales de XML
    texto_escapado = (
        texto
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Message>{texto_escapado}</Message>"
        "</Response>"
    )


def procesar_mensaje_whatsapp(numero: str, mensaje: str) -> str:
    """Procesa un mensaje de WhatsApp y devuelve TwiML con la respuesta.

    Usa el número de teléfono como session_id para mantener la memoria
    de conversación por usuario.

    Parámetros:
        numero: Número de WhatsApp del remitente (ej: "whatsapp:+573157390732").
        mensaje: Texto del mensaje del usuario.

    Devuelve:
        String TwiML con la respuesta del agente.
    """
    # Usar el número como session_id (limpiarlo para que sea un key válido)
    session_id = numero.replace("whatsapp:", "").replace("+", "").replace(" ", "")

    try:
        resultado = responder(pregunta=mensaje, session_id=session_id)
        respuesta_texto = resultado.get("respuesta", _MENSAJE_ERROR)

        # WhatsApp tiene límite de 1600 caracteres por mensaje
        if len(respuesta_texto) > 1500:
            respuesta_texto = respuesta_texto[:1497] + "..."

        logger.info(
            "Respuesta generada | session=%s | chars=%d | tool=%s",
            session_id,
            len(respuesta_texto),
            resultado.get("uso_tool_financiera", False),
        )

    except AgentError as e:
        logger.error("AgentError para %s: %s", numero, e)
        respuesta_texto = _MENSAJE_ERROR
    except Exception:
        logger.exception("Error inesperado procesando mensaje de %s", numero)
        respuesta_texto = _MENSAJE_ERROR

    return _twiml(respuesta_texto)
