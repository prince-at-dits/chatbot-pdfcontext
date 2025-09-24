import os
import torch
import gradio as gr
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from collections import Counter
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
import re


load_dotenv()


# Global states
qa_chain = None
pdf_metadata = {"name": None, "author": None, "title": None, "num_pages": 0}
pdf_entities = []
pending_entity = None
pending_answer = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
embedding_store = []
vector_db = None

# GPU device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device.upper()}")


AI_MODEL = "gpt-4o"
llm = ChatOpenAI(
    model=AI_MODEL,
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 
# Entity detection in PDF
def detect_entities(docs):
    text = " ".join([doc.page_content for doc in docs])
    pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:\s+(?:Inc|Corp|Ltd|LLC|Co|Group|Foundation|University|Hospital))?\b"
    candidates = re.findall(pattern, text)
    common_words = {"The", "This", "That", "These", "Those", "Chapter", "Section", "Figure", "Table", "Appendix"}
    filtered = [c.strip() for c in candidates if c.split()[0] not in common_words and len(c) > 2]
    counter = Counter(filtered)
    entities = [ent for ent, count in counter.most_common() if count >= 1]
    print(f"🔍 Detected entities: {entities}")
    print("======")
    return entities
print("======")

# Extract asked entity / special tokens

def extract_asked_entity(question):
    q = question.strip().lower()
    if re.search(r"\bwho (?:is|was) the author\b", q) or "author of this document" in q:
        return "__AUTHOR__"
    if re.search(r"\b(what|tell me|describe).*(this (pdf|document|file)|the (pdf|document))", q):
        return "__DOC_OVERVIEW__"

    patterns = [
        r"who is ([\w\s]+)\??$",
        r"tell me about ([\w\s]+)\??$",
        r"what (?:is|are) ([\w\s]+)\??$",
        r"info on ([\w\s]+)\??$",
        r"about ([\w\s]+)\??$",
        r"describe ([\w\s]+)\??$"
    ]
    for p in patterns:
        m = re.search(p, question, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    print("==========extract===")
    return None




# Match entities fuzzily
def match_entities(asked, entities):
    if not asked or not entities:
        return []
    a = asked.lower().strip()
    exact = [e for e in entities if a in e.lower() or e.lower() in a]
    if exact:
        return exact
    asked_tokens = set([t for t in re.findall(r"\w+", a) if len(t) > 2])
    scored = []
    for e in entities:
        tokens = set(re.findall(r"\w+", e.lower()))
        overlap = len(asked_tokens & tokens)
        scored.append((overlap, e))
    scored = sorted(scored, reverse=True)
    print("=====match entities====")
    return [e for score, e in scored if score >= 1]

# Load PDF and build embeddings
def load_pdf(pdf_file):
    global pdf_metadata, qa_chain, pdf_entities, vector_db

    pdf_metadata = {"name": os.path.basename(pdf_file.name), "author": None, "title": None, "num_pages": 0}
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()
    pdf_metadata["num_pages"] = len(documents)
    print(f"Loaded {pdf_metadata['num_pages']} pages")
    print("======load pdf=====")

    # Metadata extraction
    try:
        reader = PdfReader(pdf_file.name)
        meta = reader.metadata
        if meta:
            pdf_metadata["author"] = getattr(meta, 'author', None) or meta.get('/Author')
            pdf_metadata["title"] = getattr(meta, 'title', None) or meta.get('/Title')
    except Exception as e:
        print("Could not read PDF metadata:", e)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    pdf_entities = detect_entities(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )

    pdf_hash = str(hash(pdf_file.name + str(len(docs))))
    embeddings_path = f"./embeddings/{pdf_hash}"
    os.makedirs("./embeddings", exist_ok=True)

    if os.path.exists(embeddings_path):
        vector_db = FAISS.load_local(embeddings_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded cached embeddings from {embeddings_path}")
    else:
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(embeddings_path)
        print(f"Embeddings created and saved to {embeddings_path}")

    prompt_template = """
You are a helpful AI assistant with access to a PDF document.

Context:
{context}

Detected entities in this document: """ + (", ".join(pdf_entities) if pdf_entities else "None") + """

Instructions:
- Answer the user's question using ONLY the provided context.
- If the question is about a person or organization, focus your answer on that entity.
- If the answer is not in the context, clearly state: "I don't have enough information about that."
- Be detailed, professional, and use complete sentences. Avoid bullet points unless asked.
- Do NOT make up information.

Question: {question}
Answer:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 12}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False,
    )

    print("======load pdf end=====")
    return qa_chain

# Summarize document
def summarize_document(n_sentences=8):
    global vector_db, qa_chain
    if not vector_db or not qa_chain:
        return "No document loaded."
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 20})
    docs = retriever.get_relevant_documents("Summarize the document")
    joined = "\n\n".join([d.page_content for d in docs])
    summary_query = f"Summarize the following document into {n_sentences} sentences:\n\n{joined}"
    print("======summarize=====")
    try:
        return qa_chain.run(summary_query)
    except Exception as e:
        return f"Error summarizing: {e}"

# Chatbot logic
def chatbot_human(pdf_file, history, message, entity_choice=None, confirm=False):
    global qa_chain, memory, pending_answer, pending_entity, pdf_entities, vector_db, pdf_metadata

    msg_lower = message.lower().strip()
    answer = "I don’t have enough information to answer yet."

    # Confirm step
    if pending_answer and confirm:
        answer = f" Answer confirmed: {pending_answer}"
        pending_answer = None
        pending_entity = None
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        return history, gr.update(choices=[], interactive=False)

    asked_entity = extract_asked_entity(message)

    # Special case: author
    if asked_entity == "__AUTHOR__":
        if pdf_metadata.get("author"):
            answer = f"The author of this document is {pdf_metadata['author']}."
        else:
            answer = qa_chain.run("Who is the author of this document?") or "I don’t have enough information about the author."
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        return history, gr.update(choices=[], interactive=False)

    # Special case: full summary
    if re.search(r"\b(whole|entire|full|complete)\b.*\b(document|pdf|content|data)\b", msg_lower) or asked_entity == "__DOC_OVERVIEW__":
        answer = summarize_document(n_sentences=12)
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        return history, gr.update(choices=[], interactive=False)

    # Entity-based
    if asked_entity and pdf_entities:
        matching = match_entities(asked_entity, pdf_entities)
        if len(matching) > 1:
            pending_entity = "awaiting_selection"
            answer = f"I found multiple matches for '{asked_entity}': {', '.join(matching)}. Which one do you want?"
            history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
            return history, gr.update(choices=matching, interactive=True, value=None)
        elif len(matching) == 1:
            pending_entity = matching[0]
        else:
            answer = qa_chain.run(message) or f"I couldn't find any information about '{asked_entity}' in the document."
            history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
            return history, gr.update(choices=[], interactive=False)

    if pending_entity and pending_entity != "awaiting_selection":
        question_with_context = f"{message} [Focus on entity: {pending_entity}]"
        suggested_answer = qa_chain.run(question_with_context)
        pending_answer = suggested_answer
        answer = f"Suggested Answer: {suggested_answer}\nDo you confirm?"
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        return history, gr.update(choices=[], interactive=False)

    # General case
    if pdf_file and qa_chain:
        answer = qa_chain.run(message) or "I don’t have enough information about that."
    elif not pdf_file:
        memory.chat_memory.add_user_message(message)
        response = llm.invoke(memory.chat_memory.messages)
        answer = response.content
        memory.chat_memory.add_ai_message(answer)

    history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    print("======chatbot=====")
    return history, gr.update(choices=[], interactive=False)

# Reset chat
def new_chat():
    global qa_chain, memory, embedding_store, pdf_entities, pending_entity, pending_answer
    qa_chain = None
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    embedding_store = []
    pdf_entities = []
    pending_entity = None
    pending_answer = None
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return [], None



with gr.Blocks() as demo:
    gr.Markdown("## 📑 Local PDF + Human-Assured Chatbot (Any Text PDF)")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])

    chatbot_ui = gr.Chatbot(
        label="Chat History",
        type="messages",   # keeps previous messages
        height=400,
        bubble_full_width=False,
        avatar_images=("images.jpeg", "bot.jpg")
    )

    msg = gr.Textbox(
        label="Ask a question",
        placeholder="Type your question here..."
    )
    
    entity_dropdown = gr.Dropdown(choices=[], label="Select entity", interactive=False)
    confirm_checkbox = gr.Checkbox(label="Confirm answer")

    with gr.Row():
        clear_btn = gr.Button("Clear Chat")  # Clear chat button

    # Example questions
    gr.Examples(
        examples=[
            ["Who is the author of this document?"],
            ["Tell me about the document"],
            ["Give whole data of the pdf"],
            ["What does this report say about AI?"]
        ],
        inputs=msg,
        label="Try these example questions"
    )


    # Functions
    def update_entities(pdf_file):
        if pdf_file:
            load_pdf(pdf_file)
            return gr.update(choices=pdf_entities, interactive=True)
        return gr.update(choices=[], interactive=False)

    def submit_message(pdf_file, history, message, entity_choice=None, confirm=False):
        # Run chatbot logic
        history, entity_update = chatbot_human(pdf_file, history, message, entity_choice, confirm)
        return history, entity_update, ""  # Clear input box

    def clear_chat():
        history, _ = new_chat()
        return history, gr.update(choices=[], interactive=False), ""  # Reset dropdown and input box

    # Events
    
    pdf_file.upload(update_entities, pdf_file, entity_dropdown)
    msg.submit(submit_message, [pdf_file, chatbot_ui, msg, entity_dropdown, confirm_checkbox], 
               [chatbot_ui, entity_dropdown, msg])
    clear_btn.click(clear_chat, None, [chatbot_ui, entity_dropdown, msg])

demo.launch()








