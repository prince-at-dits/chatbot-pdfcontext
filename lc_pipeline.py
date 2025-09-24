from typing import List, Dict, Any
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS as LC_FAISS
try:
	from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
	from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader


class LCPipeline:
	def __init__(self):
		self.vector_db = None
		self.qa_chain = None
		self.entities: List[str] = []
		self.metadata: Dict[str, Any] = {"name": None, "author": None, "title": None, "num_pages": 0}

	def build_from_pdf_bytes(self, data: bytes, filename: str):
		# Write to temp, then load
		import uuid, tempfile
		tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{filename}")
		with open(tmp_path, 'wb') as f:
			f.write(data)
		try:
			# Metadata via PyPDF2
			try:
				reader = PdfReader(tmp_path)
				meta = reader.metadata
				if meta:
					self.metadata["author"] = getattr(meta, 'author', None) or meta.get('/Author')
					self.metadata["title"] = getattr(meta, 'title', None) or meta.get('/Title')
			except Exception:
				pass

			loader = PyPDFLoader(tmp_path)
			documents = loader.load()
			self.metadata["name"] = filename
			self.metadata["num_pages"] = len(documents)

			splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
			docs = splitter.split_documents(documents)

			hf_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
			if self.vector_db is None:
				self.vector_db = LC_FAISS.from_documents(docs, hf_emb)
			else:
				new_db = LC_FAISS.from_documents(docs, hf_emb)
				self.vector_db.merge_from(new_db)

			# Detect simple entities
			text_all = " ".join([d.page_content for d in docs])
			pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:\s+(?:Inc|Corp|Ltd|LLC|Co|Group|Foundation|University|Hospital))?\b"
			candidates = re.findall(pattern, text_all)
			common_words = {"The", "This", "That", "These", "Those", "Chapter", "Section", "Figure", "Table", "Appendix"}
			filtered = [c.strip() for c in candidates if c.split()[0] not in common_words and len(c) > 2]
			self.entities = sorted(set(filtered))

			AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
			llm = ChatOpenAI(model=AI_MODEL, temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
			prompt_template = """
You are a helpful AI assistant with access to a PDF document.

Context:
{context}

Detected entities in this document: """ + (", ".join(self.entities) if self.entities else "None") + """

Instructions:
- Answer the user's question using ONLY the provided context.
- If the answer is not in the context, clearly state: "I don't have enough information about that."
- Be detailed and professional.
- Do NOT make up information.

Question: {question}
Answer:
"""
			PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
			self.qa_chain = RetrievalQA.from_chain_type(
				llm=llm,
				chain_type="stuff",
				retriever=self.vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 12}),
				chain_type_kwargs={"prompt": PROMPT},
				return_source_documents=False,
			)
		finally:
			try:
				os.remove(tmp_path)
			except Exception:
				pass

	def summarize(self, n_sentences: int = 10) -> str:
		if not self.vector_db or not self.qa_chain:
			return "No document loaded."
		retriever = self.vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 20})
		docs = retriever.get_relevant_documents("Summarize the document")
		joined = "\n\n".join([d.page_content for d in docs])
		query = f"Summarize the following document into {n_sentences} sentences:\n\n{joined}"
		try:
			return self.qa_chain.run(query)
		except Exception as e:
			return f"Error summarizing: {e}"

	def ask(self, question: str) -> str:
		if not self.qa_chain:
			return "I don’t have enough information to answer yet."
		try:
			return self.qa_chain.run(question) or "I don’t have enough information about that."
		except Exception as e:
			return f"I couldn’t answer due to an internal error: {e}"
