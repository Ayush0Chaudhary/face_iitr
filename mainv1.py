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
import requests

app = FastAPI()
os.makedirs("face_db", exist_ok=True)
DB_FILE = "db.v.2.npy"
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
        embedding = DeepFace.represent(img_path=img_path, model_name='Facenet512', enforce_detection=False)[0]["embedding"]
    except Exception as e:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail=f"Face embedding failed: {e}")

    global db_entries
    # Check if user already exists
    existing_entry = next(
        (entry for entry in db_entries
        if isinstance(entry, dict) and entry.get("enrolment_number") == userId),
        None
    )
    if existing_entry:
        # Replace the existing user's photo and embedding
        device_id = "c4aac31550b5db3119f0b8cd9b0fc6a638bc813e3661faccf2a81b8b9fd345fc"
        headers = {'X-DEVICE-ID': device_id}

        res = requests.get("https://channeli.in/api/r_pravesh/extract_details/",
                        headers=headers,
                        params={"enrolment_number": str(userId)})

        existing_entry["embedding"] = embedding
        existing_entry["info"] = str(res.json())
        print(f"Updated existing user {userId} with new embedding.")
    else:
        # Add new user entry
        device_id = "c4aac31550b5db3119f0b8cd9b0fc6a638bc813e3661faccf2a81b8b9fd345fc"
        headers = {'X-DEVICE-ID': device_id}

        res = requests.get("https://channeli.in/api/r_pravesh/extract_details/",
                        headers=headers,
                        params={"enrolment_number": str(userId)})
        entry = {
            "embedding": embedding,
            "enrolment_number": userId,
            "info": str(res.json()),
        }
        print(f"New user {userId} registered with details: {res.json()}")
        print(f"New user {userId} registered with details: {res.text}")
        db_entries.append(entry)
        print(f"Registered new user {userId}.")

    # Save updated database
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
        query_embedding = DeepFace.represent(img_path=img_path, model_name='Facenet512', enforce_detection=False)[0]["embedding"]
    except Exception as e:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail=f"Face embedding failed: {e}")

    os.remove(img_path)

    # Compute similarities
    valid_entries = [entry for entry in db_entries if isinstance(entry.get("embedding"), (list, np.ndarray)) and len(entry["embedding"]) > 0]
    if not valid_entries:
        raise HTTPException(status_code=500, detail="No valid embeddings in the database.")
    print(f"Valid entries: {len(valid_entries)}")

    db_embeddings = [entry["embedding"] for entry in valid_entries]

    similarities = cosine_similarity([query_embedding], db_embeddings)[0]
    top_indices = np.argsort(similarities)[-10:][::-1]  # Get top 10 indices

    top_matches = []
    for idx in top_indices:
        matched_entry = valid_entries[idx].copy()
        matched_entry.pop("embedding")  # Do not send raw embedding in response
        matched_entry["confidence"] = float(similarities[idx])
        top_matches.append(matched_entry)

    end = time.time()
    print(f"Time taken for identification: {end - start:.2f} seconds")

    return JSONResponse({"matches": top_matches, "time_taken": end - start})



from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

# Setup Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Route to render HTML
@app.get("/app", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
