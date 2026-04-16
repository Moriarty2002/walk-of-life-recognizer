# walk-of-life-recognizer
Deep learning model to identify runner photos for Walk of Life 2026 in Naples

## How the notebook works
The main workflow is in `src/recognizer.ipynb` and is split into two parts:

1. Build embeddings database
- The notebook loads race images, detects faces with MTCNN, and extracts 512-d face embeddings with InceptionResnetV1.
- One record is saved for each detected face.

2. Search for a runner
- A query photo is processed to extract one target face embedding.
- The query embedding is compared against the database using cosine similarity.
- Top matches are returned and visualized with bounding boxes.

## How to use the notebook
Typical flow in a new environment:

1. Run setup cells (imports, Drive mount, paths, model/device setup).
2. If you need to create a new database, run Part 1 once to generate `embeddings_db.pt`.
3. Run Part 2 to load the database and search for a runner image.

### Search only (skip Part 1)
If you already have a valid `embeddings_db.pt`, you do not need to rebuild embeddings.

1. Run setup cells up to the point where paths and models are ready.
2. Do not run the `build_embedding_database(...)` execution cell.
3. In Part 2, run the database loading cell (`load_database(EMBEDDINGS_FILE)`).
4. Set your query image path in `QUERY_IMAGE`.
5. Run query embedding extraction, search, and display cells.

Note: for full visual results, the original images folder should be available locally at `IMAGES_DIR`. If images are missing, metadata search still works, but previews may not render.

## What is `embeddings_db.pt`
`src/resources/embeddings_db.pt` is a serialized PyTorch file containing the face embedding database produced by the notebook.

Each item stores:
- the source image filename
- the detected face bounding box
- the detector confidence score
- the 512-d face embedding vector

This file lets you run fast similarity search without recomputing embeddings from all photos.


# Photo source
All photos are by GARE IN FOTO.
Special thanks to the photographers Antonio, Franco, and Gabriele.
https://www.gareinfoto.com/galleria-foto-2026-1-semestre/

## Technical details
- Face detection uses `MTCNN` from `facenet-pytorch`.
- Face embeddings are extracted with `InceptionResnetV1` pretrained on `VGGFace2`.
- Each detected face is represented as a 512-dimensional embedding vector.
- Matching is done with cosine similarity, with Euclidean distance also reported for reference.
- The notebook is designed to run in a Python environment with PyTorch, and it uses Google Drive paths for input images and saved embeddings when executed in Colab.