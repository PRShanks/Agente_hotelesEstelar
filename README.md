# Hoteles Estelar — Agente Conversacional con IA (Taller 1)

Sistema completo de análisis empresarial e interacción conversacional sobre datos públicos de **Hoteles Estelar S.A.** (NIT 890304099), construido a lo largo de tres módulos progresivos.

> **Stack:** Python 3.12 · uv · LangChain · Claude (Anthropic) · OpenAI · Supabase · pgvector · FastAPI · Twilio · Railway · Streamlit

---

##  Evolución arquitectónica

### Módulo 1 — Q&A con RAG local
```
Scraper → .md → data_loader → FAISS local → Claude → Streamlit
```

### Módulo 2 — Agente conversacional con memoria
```
Scraper → Supabase (pgvector) → Agente LangChain → Memoria (LangGraph) → Streamlit
                                      ↓
                              Tool financiera (SQLite)
```

### Módulo 3 — Producción con WhatsApp
```
WhatsApp → Twilio → FastAPI (Railway) → Agente LangChain (Function Calling)
                                               ↓                ↓
                                    RAG (Supabase)    Tool financiera (SQLite)
                                               ↓
                                    Claude → Respuesta → WhatsApp
```

---

##  Estructura del proyecto

```
scraper-main/
├── api/                              # Módulo 3: API REST
│   ├── __init__.py
│   ├── main.py                       # FastAPI app con /chat y /whatsapp
│   └── whatsapp.py                   # Webhook Twilio → TwiML
├── app/
│   └── dashboard.py                  # Interfaz Streamlit (Módulo 2)
├── data/
│   ├── estelar_reportes/
│   │   ├── HOTELES_ESTELAR_890304099.md   # datos financieros (scraper)
│   │   ├── hoteles_estelar.md             # info corporativa
│   │   └── hoteles_estelar_agente_clientes.md
│   └── memoria.db                    # SQLite memoria conversacional
├── llm/
│   ├── agent.py                      # Módulo 3: Agente con Function Calling
│   ├── clients/
│   │   ├── factory.py                # Factory del LLM (Claude/Ollama)
│   │   └── memory.py                 # Memoria con LangGraph + SQLiteStore
│   ├── core/
│   │   ├── qa.py                     # Q&A con RAG + tool-calling
│   │   ├── summarizer.py             # Generador de resúmenes
│   │   └── faq_generator.py          # Generador de FAQ
│   ├── financial/
│   │   └── tool.py                   # Tool financiera (SQLite)
│   ├── models.py                     # Modelos Pydantic (RespuestaQA)
│   ├── prompts/
│   │   └── qa.py                     # System prompt del agente
│   └── rag/
│       ├── embeddings.py             # Factory de embeddings (Ollama/OpenAI)
│       ├── sanitizer.py              # Limpieza de fragmentos RAG
│       └── vector_store.py           # Cliente Supabase pgvector
├── scripts/
│   ├── extract_estelar_report.py     # Scraper Power BI (Playwright)
│   ├── extract_hotelesestelar_web.py # Scraper web oficial (BeautifulSoup)
│   ├── ingestar_supabase.py          # Carga documentos a Supabase
│   └── setup_supabase.sql            # SQL para crear tabla y función
├── tests/                            # Tests del proyecto
├── .env.example                      # Plantilla de variables de entorno
├── .gitignore
├── .python-version                   # Python 3.12
├── Makefile                          # Comandos del proyecto
├── Procfile                          # Para Railway
├── pyproject.toml                    # Dependencias con uv
├── railway.toml                      # Configuración de Railway
└── README.md
```

---

##  Instalación

Este proyecto usa **[uv](https://github.com/astral-sh/uv)** como gestor de paquetes.

### 1. Clonar el repositorio
```bash
git clone https://github.com/PRShanks/Taller1.git
cd Taller1
```

### 2. Crear entorno virtual con Python 3.12
```bash
uv venv --python 3.12
```

### 3. Activar el entorno
```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
```bash
# Mac/Linux
source .venv/bin/activate
```

### 4. Instalar dependencias
```bash
uv sync
```

### 5. Configurar variables de entorno
```powershell
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
```

Edita el `.env` con tus credenciales:
```env
# LLM
ANTHROPIC_API_KEY=sk-ant-api03-...

# Embeddings (openai para producción, ollama para local)
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Supabase (base de datos vectorial)
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=sb_secret_...

# Twilio (WhatsApp) - solo para Módulo 3
TWILIO_ACCOUNT_SID=ACxx...
TWILIO_AUTH_TOKEN=xx...
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
```

---

## ▶️ Uso

### Dashboard interactivo (Módulo 2)
```bash
make dev
# o
.venv/Scripts/python.exe -m streamlit run app/dashboard.py
```
Se abre en `http://localhost:8501`.

### API REST local (Módulo 3)
```bash
.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000
```
Documentación en `http://localhost:8000/docs`.

### Comandos del Makefile
```bash
make dev          # Lanzar dashboard Streamlit
make ingest       # Cargar documentos a Supabase (primera vez)
make reindex      # Reindexar Supabase desde cero
make test         # Ejecutar tests
make lint         # Verificar estilo de código
make clean-cache  # Limpiar __pycache__
```

---

##  Arquitectura del agente (Módulo 3)

### Function Calling estricto con Pydantic

El agente usa **Function Calling** con esquemas Pydantic que fuerzan al LLM a elegir la herramienta correcta:

```python
class BusquedaRAGInput(BaseModel):
    pregunta: str = Field(description="Consulta en lenguaje natural")
    top_k: int = Field(default=5, ge=1, le=10)
```

### Herramientas disponibles

| Herramienta | Cuándo se usa | Fuente de datos |
|---|---|---|
| `busqueda_rag` | Preguntas abiertas sobre servicios, hoteles, alianzas | Supabase pgvector |
| `query_financiero` | Preguntas sobre cifras exactas (ingresos, EBITDA, deuda) | SQLite local |

### Flujo completo
```
Usuario (WhatsApp)
      ↓
Twilio → POST /whatsapp (FastAPI)
      ↓
Agente LangChain
  1. Genera embedding de la pregunta (OpenAI, 1536 dims)
  2. Decide qué herramienta usar (Function Calling)
  3. Ejecuta la herramienta elegida
  4. Claude genera la respuesta con el resultado
      ↓
TwiML → Twilio → WhatsApp
```

---

## 🗄️ Base de datos vectorial (Supabase)

### Configurar Supabase
1. Crear proyecto en https://supabase.com
2. Ejecutar `scripts/setup_supabase.sql` en el SQL Editor
3. Configurar `SUPABASE_URL` y `SUPABASE_SERVICE_KEY` en `.env`

### Cargar documentos
```bash
make ingest
```

Carga **68 chunks** de 3 archivos de información corporativa:
- `hoteles_estelar_agente_clientes.md` — 44 chunks
- `informacion general.md` — 12 chunks
- `inteligencia empresarial.md` — 12 chunks

**Parámetros:** chunk=1000 chars, overlap=200, embeddings=1536 dims (OpenAI)

> Los reportes financieros NO se cargan al vector store — son consultados por la herramienta estructurada `query_financiero` vía SQLite.

---

##  Embeddings

| Proveedor | Modelo | Dims | Cuándo usar |
|---|---|---|---|
| OpenAI | text-embedding-3-small | 1536 | Producción (Railway) |
| Ollama | nomic-embed-text | 768 | Desarrollo local |

Configurar con `EMBEDDING_PROVIDER=openai` o `EMBEDDING_PROVIDER=ollama` en `.env`.

>  Las dimensiones deben ser consistentes entre la tabla de Supabase y el modelo. Si cambias de proveedor, ejecuta `make reindex`.

---

##  Integración WhatsApp (Módulo 3)

### Configuración de Twilio Sandbox
1. Crear cuenta en https://console.twilio.com
2. Activar sandbox: **Messaging → Try it out → Send a WhatsApp message**
3. El usuario envía `join important-onto` al **+1 415 523 8886**
4. Configurar webhook en Sandbox settings:
   ```
   https://web-production-58824.up.railway.app/whatsapp
   ```

### Deploy en Railway
El proyecto está desplegado en:
```
https://web-production-58824.up.railway.app
```

Variables de entorno configuradas en Railway:
- `ANTHROPIC_API_KEY`
- `SUPABASE_URL` + `SUPABASE_SERVICE_KEY`
- `EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY`

---

##  Módulos del proyecto

### Módulo 1 — Q&A con RAG local
- Scraping de datos financieros (Power BI / Supersociedades)
- Consolidación de datos en archivo de texto limpio
- RAG con FAISS local y Claude
- Dashboard Streamlit con Resumen, FAQ y Q&A
- Prompt engineering documentado con 3 versiones por prompt

### Módulo 2 — Agente conversacional
- Migración de FAISS a Supabase (pgvector)
- Embeddings con nomic-embed-text (Ollama, 768 dims)
- Memoria conversacional con LangGraph + SQLiteStore
- Tool financiera para datos exactos (SQLite)
- Agente enrutador con tool-calling
- Dashboard actualizado con historial de chat

### Módulo 3 — Producción
- API REST con FastAPI (`/chat` y `/whatsapp`)
- Function Calling estricto con esquemas Pydantic
- Migración de embeddings a OpenAI (1536 dims)
- Deploy en Railway (24/7)
- Integración con WhatsApp vía Twilio Sandbox

---

##  Seguridad

- `.env` en `.gitignore` — nunca se sube a GitHub
- `data/memoria.db` en `.gitignore` — datos locales de sesión
- Usar `SUPABASE_SERVICE_KEY` (secret) solo en backend, nunca en frontend
- API keys de producción configuradas como variables de entorno en Railway

---

##  Licencia

Proyecto académico — Taller 1, Aplicación de Técnicas Avanzadas de IA Generativa.
**Empresa analizada:** Hoteles Estelar S.A. — Santiago de Cali, Colombia.
