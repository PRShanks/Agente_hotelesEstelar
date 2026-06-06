"""llm/auth.py.

----------
Sistema de autenticación para acceso a información financiera sensible.

Flujo:
  1. El agente detecta que la pregunta requiere datos financieros.
  2. Verifica si la sesión ya está autenticada (en memoria).
  3. Si no, pide el código de 4 dígitos al usuario.
  4. Verifica contra la lista de empleados válidos.
  5. Si pasa, guarda el estado en memoria y da acceso.

HumanInTheLoop: el agente pausa y espera la respuesta del usuario
antes de continuar con la consulta financiera.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lista de códigos de empleados válidos
# En producción esto vendría de una base de datos
# ---------------------------------------------------------------------------
EMPLEADOS_VALIDOS = {
    "1234": "Carlos Pérez - Gerente Financiero",
    "5678": "Ana García - Directora de Operaciones",
    "9012": "Miguel Torres - Analista Financiero",
    "3456": "Laura Gómez - Contralora",
    "7890": "Juan Rodríguez - CEO",
}

# Clave en memoria para guardar el estado de autenticación
_AUTH_KEY = "auth_financiera"
_AUTH_INTENTOS_KEY = "auth_intentos"
MAX_INTENTOS = 3


def es_pregunta_financiera(pregunta: str) -> bool:
    """Detecta si una pregunta requiere acceso a datos financieros.

    Parámetros:
        pregunta: Texto de la pregunta del usuario.

    Devuelve:
        True si la pregunta es sobre datos financieros sensibles.
    """
    palabras_clave = [
        "ingreso", "ingresos", "ebitda", "utilidad", "deuda",
        "margen", "financiero", "financiera", "balance",
        "facturación", "facturacion", "revenue", "ganancia",
        "pérdida", "perdida", "cifra", "cifras", "dato financiero",
        "reporte financiero", "resultado", "resultados",
    ]
    pregunta_lower = pregunta.lower()
    return any(palabra in pregunta_lower for palabra in palabras_clave)


def esta_autenticado(memoria, session_id: str) -> bool:
    """Verifica si la sesión ya está autenticada para datos financieros.

    Parámetros:
        memoria: Instancia de SessionMemory.
        session_id: ID de la sesión.

    Devuelve:
        True si la sesión ya pasó la autenticación.
    """
    datos = memoria.get_user_data(session_id, _AUTH_KEY)
    return datos is not None and datos.get("autenticado", False)


def obtener_empleado_autenticado(memoria, session_id: str) -> str | None:
    """Obtiene el nombre del empleado autenticado en la sesión.

    Parámetros:
        memoria: Instancia de SessionMemory.
        session_id: ID de la sesión.

    Devuelve:
        Nombre del empleado o None si no está autenticado.
    """
    datos = memoria.get_user_data(session_id, _AUTH_KEY)
    if datos and datos.get("autenticado"):
        return datos.get("nombre_empleado")
    return None


def verificar_codigo(
    codigo: str,
    memoria,
    session_id: str,
) -> dict:
    """Verifica un código de 4 dígitos contra la lista de empleados.

    Parámetros:
        codigo: El código enviado por el usuario.
        memoria: Instancia de SessionMemory.
        session_id: ID de la sesión.

    Devuelve:
        Dict con:
          - 'valido': bool
          - 'mensaje': texto de respuesta al usuario
          - 'nombre_empleado': nombre si es válido
    """
    # Limpiar el código (quitar espacios)
    codigo = codigo.strip()

    # Verificar formato: exactamente 4 dígitos
    if not re.fullmatch(r"\d{4}", codigo):
        return {
            "valido": False,
            "mensaje": (
                "❌ El código debe ser exactamente 4 dígitos numéricos. "
                "Por favor intenta de nuevo."
            ),
            "nombre_empleado": None,
        }

    # Verificar intentos fallidos
    intentos_data = memoria.get_user_data(session_id, _AUTH_INTENTOS_KEY)
    intentos = intentos_data.get("intentos", 0) if intentos_data else 0

    if intentos >= MAX_INTENTOS:
        return {
            "valido": False,
            "mensaje": (
                "🚫 Has superado el número máximo de intentos. "
                "Por favor contacta al área de sistemas para desbloquear tu acceso."
            ),
            "nombre_empleado": None,
        }

    # Verificar contra la lista
    if codigo in EMPLEADOS_VALIDOS:
        nombre = EMPLEADOS_VALIDOS[codigo]

        # Guardar autenticación en memoria
        memoria.save_user_data(
            session_id,
            _AUTH_KEY,
            {"autenticado": True, "codigo": codigo, "nombre_empleado": nombre},
        )
        # Resetear intentos
        memoria.save_user_data(session_id, _AUTH_INTENTOS_KEY, {"intentos": 0})

        logger.info(
            "Autenticacion exitosa | session=%s | empleado=%s",
            session_id, nombre,
        )

        return {
            "valido": True,
            "mensaje": (
                f"✅ Autenticación exitosa. Bienvenido/a, *{nombre}*. "
                f"Ahora puedes acceder a la información financiera."
            ),
            "nombre_empleado": nombre,
        }
    else:
        # Registrar intento fallido
        memoria.save_user_data(
            session_id,
            _AUTH_INTENTOS_KEY,
            {"intentos": intentos + 1},
        )
        restantes = MAX_INTENTOS - (intentos + 1)

        logger.warning(
            "Intento fallido | session=%s | codigo=%s | intentos=%d",
            session_id, codigo, intentos + 1,
        )

        return {
            "valido": False,
            "mensaje": (
                f"❌ Código incorrecto. "
                f"Te quedan {restantes} intento(s). "
                f"Por favor verifica tu código de empleado."
            ) if restantes > 0 else (
                "🚫 Has superado el número máximo de intentos. "
                "Contacta al área de sistemas."
            ),
            "nombre_empleado": None,
        }


def mensaje_solicitud_auth() -> str:
    """Genera el mensaje que pide el código al usuario.

    Devuelve:
        Texto del mensaje de solicitud de autenticación.
    """
    return (
        "🔐 *Acceso restringido*\n\n"
        "La información financiera es confidencial y requiere autenticación.\n\n"
        "Por favor ingresa tu *código de empleado de 4 dígitos* "
        "para continuar.\n\n"
        "_Si no eres empleado de Hoteles Estelar, puedo ayudarte "
        "con información general sobre nuestros hoteles y servicios._"
    )


def cerrar_sesion(memoria, session_id: str) -> str:
    """Cierra la sesión de autenticación financiera.

    Parámetros:
        memoria: Instancia de SessionMemory.
        session_id: ID de la sesión.

    Devuelve:
        Mensaje de confirmación.
    """
    memoria.delete_user_data(session_id, _AUTH_KEY)
    memoria.delete_user_data(session_id, _AUTH_INTENTOS_KEY)
    logger.info("Sesion cerrada | session=%s", session_id)
    return "✅ Sesión cerrada. Tu acceso financiero ha sido revocado."
