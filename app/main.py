from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Chat ADJ API")

# Conexão com pgvector
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="participantes_adj",
    connection=os.getenv("DATABASE_URL"),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Históricos por sessão
historicos = {}

def get_historico(session_id: str):
    if session_id not in historicos:
        historicos[session_id] = ChatMessageHistory()
    return historicos[session_id]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um assistente da Igreja ADJ.
Responda de forma cristã, com emojis, sem gírias.
Use APENAS as informações abaixo para responder.
Se não encontrar, diga: 'Não localizei esse dado na base. 🙏'

Informações encontradas:
{contexto}"""),
    ("placeholder", "{historico}"),
    ("human", "{pergunta}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def formatar_contexto(docs):
    return "\n".join([doc.page_content for doc in docs])

def buscar_contexto(input):
    docs = retriever.invoke(input["pergunta"])
    return formatar_contexto(docs)

# Chain com memória
chain = RunnableWithMessageHistory(
    {
        "contexto": RunnableLambda(buscar_contexto),
        "pergunta": lambda x: x["pergunta"]
    }
    | prompt
    | llm,
    get_historico,
    input_messages_key="pergunta",
    history_messages_key="historico",
)

# Models
class ChatRequest(BaseModel):
    session_id: str
    pergunta: str

class ChatResponse(BaseModel):
    resposta: str

# Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat/adj", response_model=ChatResponse)
async def chat_adj(req: ChatRequest):
    try:
        config = {"configurable": {"session_id": req.session_id}}
        resposta = chain.invoke({"pergunta": req.pergunta}, config=config)
        return ChatResponse(resposta=resposta.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))