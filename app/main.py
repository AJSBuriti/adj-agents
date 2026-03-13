from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy import create_engine, text
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
engine = create_engine(os.getenv("DATABASE_URL"))
with engine.connect() as conn:
    total = conn.execute(text("SELECT COUNT(*) FROM participantes_adj WHERE ativo = true")).scalar()
retriever = vectorstore.as_retriever(search_kwargs={"k": total})

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

def calcular_idade(data_nascimento: str | None) -> str:
    if not data_nascimento:
        return "Idade desconhecida"
    try:
        from datetime import date
        nascimento = date.fromisoformat(str(data_nascimento)[:10])
        hoje = date.today()
        idade = hoje.year - nascimento.year - ((hoje.month, hoje.day) < (nascimento.month, nascimento.day))
        return f"{idade} anos"
    except:
        return "Idade desconhecida"

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

class SyncRequest(BaseModel):
    id: int
    nome: str
    cidade: str | None = None
    estado: str | None = None
    data_nascimento: str | None = None
    igreja: str | None = None
    departamentos: str | None = None

class DeleteRequest(BaseModel):
    id: int

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

@app.post("/embeddings/sync")
async def sync_embedding(req: SyncRequest):
    try:
        # Monta o texto igual ao da indexação original
        idade = calcular_idade(req.data_nascimento)
        texto = f"Nome: {req.nome}. Idade: {idade}."
        if req.cidade:
            texto += f" Cidade: {req.cidade}"
            if req.estado:
                texto += f", {req.estado}."
            else:
                texto += "."
        if req.igreja:
            texto += f" Igreja: {req.igreja}."
        if req.departamentos:
            texto += f" Departamentos: {req.departamentos}."

        metadado = {"participante_id": req.id, "nome": req.nome}

        # Remove embedding antigo do participante se existir
        vectorstore.delete(ids=[str(req.id)])

        # Adiciona embedding novo
        vectorstore.add_texts(texts=[texto], metadatas=[metadado], ids=[str(req.id)])

        # Atualiza o k do retriever
        global retriever
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM participantes_adj WHERE ativo = true")).scalar()
        retriever = vectorstore.as_retriever(search_kwargs={"k": total})

        return {"status": "ok", "participante": req.nome, "texto_indexado": texto}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings/delete")
async def delete_embedding(req: DeleteRequest):
    try:
        # Remove o embedding pelo participante_id nos metadados
        with engine.connect() as conn:
            result = conn.execute(text("""
                DELETE FROM langchain_pg_embedding
                WHERE cmetadata->>'participante_id' = :pid
            """), {"pid": str(req.id)})
            conn.commit()
            deleted = result.rowcount

        # Atualiza o k do retriever
        with engine.connect() as conn:
            total = conn.execute(
                text("SELECT COUNT(*) FROM participantes_adj WHERE ativo = true")
            ).scalar()

        global retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": max(1, total)})

        return {"status": "deleted", "participante_id": req.id, "rows_deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))