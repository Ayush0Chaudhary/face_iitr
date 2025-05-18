# import os
# import io
# import sqlite3
# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, UploadFile, File, Form
# from deepface import DeepFace
# import faiss

# app = FastAPI()

# EMBEDDING_DIM = 128
# FAISS_INDEX_PATH = "embeddings.index"
# SQLITE_DB_PATH = "metadata.db"

# # ========== Initialize FAISS ==========
# if os.path.exists(FAISS_INDEX_PATH):
#     index = faiss.read_index(FAISS_INDEX_PATH)
# else:
#     index = faiss.IndexFlatIP(EMBEDDING_DIM)

# # ========== Initialize SQLite ==========
# conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute('''CREATE TABLE IF NOT EXISTS users (
#     userId TEXT PRIMARY KEY,
#     name TEXT,
#     email TEXT,
#     phone TEXT,
#     bhawan TEXT,
#     room TEXT
# )''')
# conn.commit()

# # ========== Helpers ==========
# def extract_embedding(file: UploadFile) -> np.ndarray:
#     img = Image.open(io.BytesIO(file.file.read()))
#     img.save("temp.jpg")  # DeepFace needs a path
#     embedding = DeepFace.represent(img_path="temp.jpg", model_name="Facenet")[0]["embedding"]
#     vector = np.array(embedding).astype("float32").reshape(1, -1)
#     faiss.normalize_L2(vector)
#     return vector

# def get_user_metadata(userId: str):
#     cursor.execute("SELECT * FROM users WHERE userId=?", (userId,))
#     return cursor.fetchone()

# # ========== Register ==========
# @app.post("/register")
# async def register_face(
#     userId: str = Form(...),
#     name: str = Form(...),
#     email: str = Form(...),
#     phone: str = Form(...),
#     bhawan: str = Form(...),
#     room: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     embedding = extract_embedding(file)

#     # Delete existing from FAISS and DB if exists
#     cursor.execute("SELECT rowid FROM users WHERE userId=?", (userId,))
#     row = cursor.fetchone()
#     if row:
#         index.remove_ids(np.array([row[0]]))  # remove from FAISS
#         cursor.execute("DELETE FROM users WHERE userId=?", (userId,))

#     index.add(embedding)
#     cursor.execute("INSERT INTO users (userId, name, email, phone, bhawan, room) VALUES (?, ?, ?, ?, ?, ?)",
#                    (userId, name, email, phone, bhawan, room))
#     conn.commit()
#     faiss.write_index(index, FAISS_INDEX_PATH)
#     return {"status": "registered", "userId": userId}

# # ========== Identify ==========
# @app.post("/identify")
# async def identify_face(file: UploadFile = File(...)):
#     query = extract_embedding(file)
#     D, I = index.search(query, k=1)
#     idx = I[0][0]
#     score = float(D[0][0])

#     if score < 0.75:  # cosine threshold
#         return {"status": "no match", "score": score}

#     cursor.execute("SELECT userId, name FROM users LIMIT 1 OFFSET ?", (idx,))
#     user = cursor.fetchone()
#     return {
#         "status": "match",
#         "score": score,
#         "userId": user[0],
#         "name": user[1]
#     }


import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
import time

app = FastAPI()
os.makedirs("face_db", exist_ok=True)
DB_FILE = "db.npy"

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
        query_embedding = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
    except Exception as e:
        os.remove(img_path)
        raise HTTPException(status_code=400, detail=f"Face embedding failed: {e}")

    os.remove(img_path)

    # Compute similarities
    db_embeddings = [entry["embedding"] for entry in db_entries]
    similarities = cosine_similarity([query_embedding], db_embeddings)[0]
    top_idx = int(np.argmax(similarities))

    matched_entry = db_entries[top_idx].copy()
    matched_entry.pop("embedding")  # Do not send raw embedding in response
    matched_entry["confidence"] = float(similarities[top_idx])

    end = time.time()
    matched_entry["time_taken"] = end - start
    print(f"Time taken for identification: {end - start:.2f} seconds")

    return JSONResponse(matched_entry)


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
        query_embedding = DeepFace.represent(img_path=img_path, model_name='Facenet512', )[0]["embedding"]
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
