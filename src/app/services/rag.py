# src/app/services/rag.py
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
import asyncio
from dotenv import load_dotenv

class RAGService:
    def __init__(self, docs_path: str = "docs", persist_dir: str = ".chroma", max_history: int = 8):
        load_dotenv()
        self.emb = OpenAIEmbeddings()                          
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.2)

        loader = DirectoryLoader(Path(docs_path), glob="**/*.md")
        docs = loader.load()                                     

        settings = Settings(anonymized_telemetry=False,          
                            persist_directory=persist_dir)
        self.vectordb = Chroma.from_documents(
            docs, self.emb, client_settings=settings
        )                                                       

        self._chains = {}
        self._max_history = max_history

        self._reply_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=(
                "Eres el asistente oficial de ClaraAI.\n\n"
                "Historial reciente:\n{chat_history}\n"
                "Contexto de la base de conocimiento:\n{context}\n\n"
                "Instrucciones:\n"
                "1. Si el historial incluye un nombre propio, saluda al usuario por su nombre.\n"
                "2. Responde en español con ≤ 4 frases concisas.\n"
                "3. Cita las fuentes como «[docs/…]» al final.\n"
                "4. Si no hay contexto relevante responde: "
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
        top_hit = self.vectordb.similarity_search_with_score(question, k=1)
        if not top_hit:                                  
            return "Lo siento, no tengo información sobre eso.", []

        doc0, dist0 = top_hit[0]                         
        sim0 = 1 - dist0                                 

        if sim0 < τ:                                 
            return "Lo siento, no tengo información sobre eso.", []

        chain  = self._get_chain(uid)
        result = chain.invoke({"question": question})

        answer  = result["answer"]
        sources = [d.metadata["source"] for d in result["source_documents"]]
        return answer, sources

    async def ask_stream(self, question: str, uid: str, τ: float = 0.15):
        """Async generator that yields the answer token-by-token."""
        print(">> ask_stream called:", question)
        history = self._get_chain(uid).memory.load_memory_variables({})["chat_history"]
        print("📝 chat_history:", history or "(empty)")
        top_hit = self.vectordb.similarity_search_with_score(question, k=1)
        if not top_hit or 1 - top_hit[0][1] < τ:
            yield "Lo siento, no tengo información sobre eso."
            return

        cb_answer = AsyncIteratorCallbackHandler()          

        llm_stream = ChatOpenAI(                            
            model_name="gpt-4.1-nano",
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

        task = asyncio.create_task(chain.ainvoke({"question": question}))

        async for token in cb_answer.aiter():
            yield token                     

        result  = await task                
        sources = [d.metadata["source"] for d in result["source_documents"]]
        if sources:
            yield f"\n\nFuentes: {', '.join(sources)}"

    def _smart_retriever(self, k: int = 3):
        return self.vectordb.as_retriever(
            search_type="similarity",               
            search_kwargs={"k": k}
        )