import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
import time
import faiss

app = FastAPI()
os.makedirs("face_db", exist_ok=True)
DB_FILE = "db.npy"
FAISS_INDEX_PATH = "faiss.index"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load or initialize database
db_entries = np.load(DB_FILE, allow_pickle=True).tolist() if os.path.exists(DB_FILE) else []

# Collect all embeddings first
all_embeddings = np.array([entry["embedding"] for entry in db_entries], dtype=np.float32)
embedding_dim = all_embeddings.shape[1]

if not os.path.exists(FAISS_INDEX_PATH):
    # Initialize FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    # Add ALL embeddings at once
    index.add(all_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
else:
    # Load existing FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)


@app.post("/register")
async def register_face(
    userId: str = Form(...),
    file: UploadFile = File(...)
):
    global index, db_entries
    
    # Save image
    img_path = f"face_db/{uuid4().hex}_{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Compute embedding
    try:
        embedding = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
        embedding_np = np.array([embedding], dtype=np.float32)  # Convert to numpy and reshape for FAISS
    except Exception as e:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail=f"Face embedding failed: {e}")
    
    # Create entry for database
    entry = {
        "embedding": embedding,
        "enrolment_number": "enrolment_number",
        "name": userId,
        "phone_number": "phone_number",
        "email_id": "email_id",
        "bhawan": "bhawan",
        "room_number": "room_number",
        "identification_key": "identification_key",
        "display_picture_path": img_path
    }
    
    # Update database and save
    db_entries.append(entry)
    np.save(DB_FILE, db_entries)
    
    # Update FAISS index
    if index is None:
        # Create new index if none exists
        embedding_dim = len(embedding)
        index = faiss.IndexFlatL2(embedding_dim)  # Changed to L2 for consistency
        
    # Add the new embedding to the index
    index.add(embedding_np)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Saved to FAISS Index")
    
    return JSONResponse({"message": "User registered successfully", "userid": userId})

@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    # Load latest DB
    global index

    print("Inside identify")
    print(f"File name: {file.filename}")
    start = time.time()
    # Check if DB exists
    if not os.path.exists(DB_FILE):
        raise HTTPException(status_code=400, detail="No face database available.")
    

    db_entries = np.load(DB_FILE, allow_pickle=True).tolist()

    if not db_entries:
        raise HTTPException(status_code=400, detail="No faces registered.")

    # Save query image temporarily
    img_path = f"temp_{uuid4().hex}_{file.filename}"
    print(f"Image path: {img_path}")
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        print(f"Writing to {img_path}")

    # Get embedding
    try:
        print(f"Computing embedding for {img_path}")
        query_embedding = np.array(DeepFace.represent(img_path=img_path, model_name='Facenet', detector_backend="retinaface")[0]["embedding"])
        print(f"Query embedding: {query_embedding}")
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1) # faiss needs float32

    except Exception as e:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail=f"Face embedding failed: {e}")

    os.remove(img_path)

    # Compute similarities
    similarities, top_idx = index.search(query_embedding, k=10) 
    similarities = similarities[0]
    top_idx = top_idx[0]

    print(f"Top indices: {top_idx}")
    print(f"Similarities: {similarities}")

    matched_entries = []
    for idx in list(top_idx):
        entry = db_entries[idx].copy()
        entry.pop("embedding") # Do not send raw embedding in response
        truncated_idx = np.where(top_idx == idx)[0][0]
        entry["confidence"] = float(similarities[truncated_idx]/100)
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
