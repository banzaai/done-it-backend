import truststore
truststore.inject_into_ssl()  # use Windows cert store — fixes enterprise SSL inspection

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import os

from schema import ChatRequest, ChatResponse, UploadDocumentResponse
from database import ConversationManager
from llm import Startup

app = FastAPI(title="Done-it Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_manager = ConversationManager()
startup = Startup()

LANG_NAMES = {"nl": "Dutch", "fr": "French", "en": "English", "es": "Spanish"}


@app.post("/chat/{conversation_id}", response_model=ChatResponse)
async def chat(conversation_id: str, request: ChatRequest):
    # Store user message
    await conversation_manager.add_message(conversation_id, "user", request.messages)

    # Build messages list for the model
    prev = await conversation_manager.get_conversation(conversation_id)
    lang_name = LANG_NAMES.get(request.lang, "Dutch")
    system_prompt = f"You are a helpful assistant for Done-it, a Belgian GPS time-tracking and project management app. Always respond in {lang_name}."
    messages = [{"role": "system", "content": system_prompt}]
    for msg in prev:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Call model (sync wrapped — InferenceClient is synchronous)
    response_text = startup.chat(messages)

    # Store and return
    await conversation_manager.add_message(conversation_id, "assistant", response_text)
    return ChatResponse(response=response_text)


@app.post("/chat/upload_document", response_model=UploadDocumentResponse)
async def upload_document(conversation_id: str = Form(...), file: UploadFile = File(...)):
    UPLOAD_PATH = "data/uploads/"
    os.makedirs(UPLOAD_PATH, exist_ok=True)
    doc_id = str(uuid.uuid4())

    try:
        content = await file.read()
        text_content = content.decode("utf-8")

        filepath = os.path.join(UPLOAD_PATH, f"{doc_id}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_content)

        chunks_added = startup.add_document_to_index(text_content, doc_id, source=file.filename)

        await conversation_manager.add_message(
            conversation_id, "system",
            f"Uploaded document: {file.filename} ({chunks_added} chunks indexed)"
        )

        return UploadDocumentResponse(
            document_id=doc_id, filename=file.filename, path=filepath, status="indexed"
        )
    except Exception as e:
        return UploadDocumentResponse(
            document_id=doc_id, filename=file.filename, path="", status=f"error: {str(e)}"
        )


@app.post("/chat/index_document")
def index_document(doc_id: str):
    try:
        with open(f"data/uploads/{doc_id}.txt", "r") as f:
            content = f.read()
        startup.add_document_to_index(content, doc_id)
    except FileNotFoundError:
        return {"error": "Document not found"}


@app.get("/chat/query")
def query_chat(conversation_id: str, user_message: str):
    relevant_chunks = startup.query_index(user_message)
    prompt = f"User message: {user_message}\nRelevant information:\n"
    for chunk in relevant_chunks:
        prompt += f"- {chunk['text']}\n"
    response_text = startup.chat([{"role": "user", "content": prompt}])
    return {"response": response_text}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

