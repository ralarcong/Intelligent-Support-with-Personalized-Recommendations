# src/app/services/rag.py
"""
rag
===

High-level wrapper around a Retrieval-Augmented Generation (RAG) stack
used by ClaraAI:

* Embedding model  – ``OpenAIEmbeddings`` (text-embedding-3-small).
* Vector store     – local **Chroma** collection (persistent).
* LLM              – ``ChatOpenAI`` (gpt-4.1-mini) combined via
  LangChain’s ``ConversationalRetrievalChain``.
* Streaming        – token-level SSE through :py:meth:`ask_stream`.

The class also tracks per-user short-term memory, mood detection and
offers a basic off-scope filter to reject requests that are not related
to ClaraAI’s product/knowledge base.

Environment
-----------
The OpenAI key **must** be available as ``OPENAI_API_KEY`` (loaded by
*python-dotenv* before any client instantiation).
"""
from pathlib import Path
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_core.tools import Tool
from app.tools.mood import detect_mood
from collections import defaultdict
from pprint import pprint
import asyncio
import re
from dotenv import load_dotenv

class RAGService:
    """
    Retrieval-Augmented Q&A service for ClaraAI.

    Parameters
    ----------
    docs_path : str, default ``"docs"``
        Folder containing knowledge-base Markdown files.
    persist_dir : str, default ``".chroma"``
        Disk location of the Chroma collection (created if missing).
    max_history : int, default 8
        Maximum number of past user/assistant messages kept in the
        :class:`langchain.memory.ConversationBufferMemory`.

    Attributes
    ----------
    vectordb : chromadb.api.models.Collection
        Shared vector store with embedded KB chunks.
    emb : langchain_openai.OpenAIEmbeddings
        Embedding client initialised with the active OpenAI key.
    llm : langchain_openai.ChatOpenAI
        LLM used for question re-phrasing and answer generation.
    _reply_template : langchain.prompts.PromptTemplate
        Prompt enforcing brevity (≤ 4 Spanish sentences) and including
        hidden chain-of-thought instructions.
    """
    def __init__(self, docs_path: str = "docs", persist_dir: str = ".chroma", max_history: int = 8):
        load_dotenv()
        self.emb = OpenAIEmbeddings()                          
        self.llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.2)
        self.llm_tools = self.llm.bind_tools([detect_mood])
        
        loader = DirectoryLoader(Path(docs_path), glob="**/*.md")
        docs = loader.load()  

        for doc in docs:                           # doc = langchain.schema.Document
            # docs/payments/fees.md  →  "payments"
            doc.metadata["topic"] = Path(doc.metadata["source"]).parts[1]                                   

        settings = Settings(anonymized_telemetry=False,          
                            persist_directory=persist_dir)
        self.vectordb = Chroma.from_documents(
            docs, self.emb, client_settings=settings
        )                                                       

        self._chains = {}
        self._max_history = max_history
        self._user_mood = defaultdict(lambda:{"style": "profesional", "emoji": "🙂"}
        )

        self._reply_template = PromptTemplate(
            input_variables=["context", "question", "chat_history", "style", "emoji"],
            template=(
                "Eres el asistente oficial de ClaraAI {emoji}.\n\n"
                "Historial reciente:\n{chat_history}\n"
                "Contexto de la base de conocimiento:\n{context}\n\n"
                "Instrucciones (tu tono debe ser {style}):\n"
                "1. Si el historial incluye un nombre propio, saluda al usuario por su nombre.\n"
                "2. Responde en español con ≤ 4 frases concisas.\n"
                "3. Si no hay contexto relevante responde: "
                "\"Lo siento, no tengo información sobre eso\".\n\n"
                "Pregunta:\n{question}\n"
            ),
        )


    def _get_chain(self, uid: str):
        if uid not in self._chains:
            mem = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=True,
                max_token_limit=self._max_history,
            )

            qa_chain = self.llm | RunnablePassthrough()  # just to keep LCEL syntax

            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self._smart_retriever(),
                memory=mem,
                combine_docs_chain_kwargs={"prompt": self._reply_template},
                return_source_documents=True,
            )
            self._chains[uid] = chain
        return self._chains[uid]

    def ask(self, question: str, uid: str, τ: float = 0.15):
        """
        Sequential approach
        """
        self._update_mood(uid, question)
        top_hit = self.vectordb.similarity_search_with_score(question, k=1)
        if not top_hit:                                  
            return "Lo siento, no tengo información sobre eso.", []

        doc0, dist0 = top_hit[0]                         
        sim0 = 1 - dist0                                 

        if sim0 < τ:                                 
            return "Lo siento, no tengo información sobre eso.", []

        vars_in = {
            "question": question,
            "style":   self._user_mood[uid]["style"],
            "emoji":   self._user_mood[uid]["emoji"],
        }
        result = self._get_chain(uid).invoke(vars_in)

        answer  = result["answer"]
        sources = [d.metadata["source"] for d in result["source_documents"]]
        return answer, sources
    
    

    def _is_off_scope(self, text: str) -> bool:
        OFF_SCOPE_PATTERNS = [r"<html", r"código html", r"javascript", r"css",r"poema", r"chiste", r"receta", r"meme",]
        return any(re.search(p, text, re.I) for p in OFF_SCOPE_PATTERNS)

    async def ask_stream(self, question: str, uid: str, τ: float = 0.15):
        """
        Server-Sent Events generator that yields answer tokens.

        The same similarity guard and off-scope filter are applied as
        in :pymeth:`ask`.

        Yields
        ------
        str
            Token or partial chunk emitted by the streaming LLM.
        """
        print(">> ask_stream called:", question)
        if self._is_off_scope(question):
            yield "Lo siento, no puedo ayudar con eso."
            return

        self._update_mood(uid, question)
        history = self._get_chain(uid).memory.load_memory_variables({})["chat_history"]
        print("📝 chat_history:", history or "(empty)")
        top_hit = self.vectordb.similarity_search_with_score(question, k=1)
        if not top_hit or 1 - top_hit[0][1] < τ:
            yield "Lo siento, no tengo información sobre eso."
            return

        cb_answer = AsyncIteratorCallbackHandler()          

        llm_stream = ChatOpenAI(                            
            model_name="gpt-4.1-mini",
            temperature=0.2,
            streaming=True,
            callbacks=[cb_answer],
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm_stream,                
            condense_question_llm=self.llm, 
            retriever=self._smart_retriever(),
            memory=self._get_chain(uid).memory,
            combine_docs_chain_kwargs={     
                "prompt": self._reply_template,
            },
            return_source_documents=True,
        )

        task = asyncio.create_task(
            chain.ainvoke({
                "question": question,
                "style":   self._user_mood[uid]["style"],
                "emoji":   self._user_mood[uid]["emoji"],
            })
        )

        async for token in cb_answer.aiter():
            yield token                     

        result  = await task                
        sources = [d.metadata["source"] for d in result["source_documents"]]
        print("SOURCES",sources)
        if sources:
            payload = ", ".join(sources)
            yield f"\n Sources \ndata: {payload}\n\n"

    def _smart_retriever(self, k: int = 3):
        return self.vectordb.as_retriever(
            search_type="similarity",               
            search_kwargs={"k": k}
        )
        
    def _update_mood(self, uid: str, text: str):
        mood = detect_mood.run(text)            # ejecuta la tool aquí mismo
        self._user_mood[uid] = {
            "style": mood["style"],
            "emoji": mood["emoji"],
        }
        print(f"[mood] {uid} → {mood}")         # 👀 visible en terminal


