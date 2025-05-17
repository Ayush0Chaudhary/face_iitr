import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()
os.makedirs("face_db", exist_ok=True)
DB_FILE = "db.npy"
N_MATCHES = 10
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load or initialize database
db_entries = np.load(DB_FILE, allow_pickle=True).tolist() if os.path.exists(DB_FILE) else []

@app.post("/register")
async def register_face(
    userId: str = Form(...),
    file: UploadFile = File(...)
):
    # Save image
    img_path = f"face_db/{uuid4().hex}_{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Compute embedding
    try:
        embedding = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
    except Exception as e:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail=f"Face embedding failed: {e}")

    entry = {
        "embedding": embedding,
        "enrolment_number": "enrolment_number",
        "name": userId,
        "phone_number": "phone_number",
        "email_id": "email_id",
        "bhawan": "bhawan",
        "room_number": "room_number",
        "identification_key": "identification_key",
        "display_picture_path": "img_path"
    }

    global db_entries
    db_entries.append(entry)
    np.save(DB_FILE, db_entries)

    return JSONResponse({"message": "User registered successfully", "userid": userId})

@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    # Load latest DB
    start = time.time()
    if not os.path.exists(DB_FILE):
        raise HTTPException(status_code=400, detail="No face database available.")

    db_entries = np.load(DB_FILE, allow_pickle=True).tolist()

    if not db_entries:
        raise HTTPException(status_code=400, detail="No faces registered.")

    # Save query image temporarily
    img_path = f"temp_{uuid4().hex}_{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get embedding
    try:
        query_embedding = DeepFace.represent(img_path=img_path, model_name='Facenet', )[0]["embedding"]
    except Exception as e:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail=f"Face embedding failed: {e}")

    os.remove(img_path)

    # Compute similarities
    db_embeddings = [entry["embedding"] for entry in db_entries]
    similarities = cosine_similarity([query_embedding], db_embeddings)[0]
    top_idx = np.argsort(similarities)[::-1][:N_MATCHES] # Selects top 10 resuls

    matched_entries = []
    for idx in top_idx:
        entry = db_entries[idx].copy()
        print(type(entry))
        entry.pop("embedding") # Do not send raw embedding in response
        entry["confidence"] = float(similarities[idx])
        matched_entries.append(entry)

    end = time.time()
    result = {
        "total_matches": len(matched_entries),
        "total_entries": len(db_entries),
        "time_taken": end - start,
        "matches": matched_entries,
    }
    print(f"Time taken for identification: {end - start:.2f} seconds")

    return JSONResponse(result)
