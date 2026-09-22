# Simple AI Chatbot (Streamlit)

A small intent-based chatbot built as a machine learning class project.

## How it works
1. `intents.json` holds example phrases (patterns) grouped under tags like
   `greeting`, `joke`, `goodbye`, `study_tip`, etc., each with a few possible
   replies.
2. `chatbot_utils.py` turns those patterns into a training set, splits it
   into train/test, and trains a **TF-IDF + Logistic Regression** classifier
   to predict the tag of any new message. Accuracy is measured on the test
   split, then the final model is retrained on all patterns for actual use.
3. `app.py` is the Streamlit chat interface: it predicts the tag of your
   message and replies with a random response from that tag (or a fallback
   reply if the model isn't confident).

## Files
- `app.py` — Streamlit chat app
- `chatbot_utils.py` — data loading, training, prediction (no Streamlit import)
- `intents.json` — training data (tags, patterns, responses)
- `requirements.txt` — Python packages needed

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Upload to GitHub aur Streamlit par live deploy karna (step by step)

### Part 1 — GitHub par upload
1. https://github.com par login karein.
2. Top-right par **+** icon → **New repository** par click karein.
3. Repository ka naam dein, jaise `simple-ai-chatbot` — **Public** rakhein
   (Streamlit free hosting ke liye public repo chahiye).
4. **Create repository** dabayein (README/gitignore add karne ki zaroorat
   nahi, hum apni files khud upload karenge).
5. Naye khaali repo ke page par **"uploading an existing file"** link par
   click karein.
6. Yeh 4 files drag-and-drop karein (ya "choose your files" se select
   karein):
   - `app.py`
   - `chatbot_utils.py`
   - `intents.json`
   - `requirements.txt`
7. Neeche **Commit changes** button dabayein. Ab saari files GitHub repo
   mein aa jayengi.

### Part 2 — Streamlit Community Cloud par deploy karna
1. https://share.streamlit.io par jayein aur apne GitHub account se login
   karein (agar pehli baar hai to Streamlit ko GitHub access dene ki
   permission dena hogi).
2. **"New app"** / **"Create app"** button par click karein.
3. **Repository** wale dropdown mein apna repo select karein
   (`your-username/simple-ai-chatbot`).
4. **Branch**: `main` select karein.
5. **Main file path**: `app.py` likhein.
6. **Deploy** button dabayein.
7. Pehli baar deploy hote waqt Streamlit `requirements.txt` se packages
   install karega — is mein 1-2 minute lag sakte hain. Uske baad app ka
   live link mil jayega, jaisay:
   `https://your-username-simple-ai-chatbot.streamlit.app`

### Agar app mein koi error aaye
- Sab se pehle check karein ke `intents.json` bilkul isi naam se, `app.py`
  ke sath usi folder (repo root) mein maujood hai.
- Streamlit Cloud ke "Manage app" page par **"View logs"** se exact error
  dekha ja sakta hai.
- Agar package error aaye, `requirements.txt` mein version numbers hata
  kar sirf naam likh dein (`streamlit`, `scikit-learn`, `pandas`, `numpy`)
  aur dobara deploy karein.

## Note on model accuracy
The training set is small (110 example phrases across 17 tags), so this is
meant to demonstrate the ML pipeline (train/test split, classifier,
prediction) rather than to be a production chatbot. It handles the phrases
it was trained on well; for something completely unrelated (e.g. a general
knowledge question) it may occasionally guess the wrong tag or fall back to
"I didn't understand that" — you can add more patterns to `intents.json` to
improve coverage.
