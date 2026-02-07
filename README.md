# Neural Two-Tower Movie Recommender

A deep learning-based recommendation system built using a **Two-Tower architecture**. This project leverages the MovieLens 100k dataset to provide personalized movie suggestions by learning user preferences and item characteristics in a joint latent space.

---

## Tech Stack
* **Language:** Python 3.x
* **Framework:** TensorFlow / Keras
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib
* **Environment:** WSL (Ubuntu) / Linux

---

## Model Architecture
The core of this project is a **Two-Tower Neural Network**. Unlike simple collaborative filtering, this architecture maps both users and movies into the same 32-dimensional vector space.

### 1. User Tower
* **Input:** `userId`
* **Layers:** Embedding(32) → Flatten → Dense(64, ReLU) → Dense(32, Linear)
* **Goal:** Create a "User Vector"

### 2. Movie Tower
* **Input:** `movieId` + `genre_features` (Multi-label one-hot encoded)
* **Layers:** Embedding(32) + Concatenation → Dense(64, ReLU) → Dense(32, Linear)
* **Goal:** Create a "Movie Vector"

### 3. Interaction
* The model computes the **Dot Product** between the User and Movie vectors to predict the user's likely rating (1-5 stars).

---

## Key Features
* **Top-K Recommendation:** A custom retrieval function that generates embeddings for all 1,600+ movies and identifies the top matches for any user.
* **Metadata Integration:** By using movie genres, the system helps mitigate the "cold-start" problem for items with fewer ratings.
