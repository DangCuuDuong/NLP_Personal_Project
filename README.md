# NLP PROCESS – Streamlit Application

> **Multifunctional NLP & Recommender demo** – Crawl & label Amazon reviews, clean & augment text, vectorize, benchmark classic ML classifiers, chat with an OpenAI‑powered assistant, and get personalised movie suggestions – all from a single Streamlit UI.

---

## ✨ Features

| Tab                                                                                                                                   
| ------------------------------------------------------------------------------------------------------------------------------------- 
| **🚀 Huấn luyện & Dự đoán**                      
|• Crawl Amazon product reviews from a URL |                                                                     
| • Upload your own CSV                                                                                                                
| • Merge, preview & download combined data                                                                                             
| • One‑click text‑cleaning pipeline (contractions → spell‑check → stopwords → stemming/lemmatise → emoji, POS, NER …)                  
| • Data augmentation with 8 classic NLP tricks (insert, substitute, keyboard noise, back‑translate, etc.)                              
| • 6 vectorisers (One‑hot, BoW, N‑gram, TF‑IDF, Word2Vec, FastText)                                                                    
| • Train & compare **K‑NN, Decision Tree, SVM, Logistic Regression**; confusion‑matrices + classification reports + accuracy bar chart 
| • Instant prediction with the best model                                                                                             
| **🤖 Chatbot**                                                                                                  
| Chat in real time with **DUONGPT** – an OpenAI‑powered Vietnamese assistant (history preserved per session)   
| **🎬 Movies recommendations**                                                                                                                   
| Rate a few movies and receive top‑N recommendations via item‑based cosine similarity on the MovieLens dataset 
---

## 🗂 Project structure

```text
├── main.py               # Streamlit UI – the entry point
├── crawl.py              # Crawl Amazon reviews (BeautifulSoup & requests)
├── CleanData.py          # Tokenisation, stop‑word removal, stemming, lemmatisation, emoji, POS, NER, …
├── GenData.py            # Text‑augmentation helpers
├── vectorize.py          # OneHot/BoW/N‑Gram/TF‑IDF/Word2Vec/FastText wrappers
├── training.py           # Model‑specific train_* functions
├── movies.csv, ratings.csv
└── requirements.txt      # Python dependencies
```

> **Note:** Only `main.py` is provided here; the other modules follow the naming shown above. Feel free to adapt paths / functions if your local modules differ.

---

## ⚙️ Installation & setup

1. **Clone & cd**

   ```bash
   git clone https://github.com/your‑user/nlp‑process‑app.git
   cd nlp‑process‑app
   ```

2. **Create virtual env** (optional but recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
   ```

3. **Install Python libraries**

   ```bash
   pip install -r requirements.txt
   # For spaCy models
   python -m spacy download en_core_web_sm
   ```

4. **Add your OpenAI key**

   ```bash
   export OPENAI_API_KEY="sk‑..."            # Linux/macOS
   setx OPENAI_API_KEY "sk‑..."              # Windows PowerShell (permanent)
   ```

5. **Run the app**

   ```bash
   streamlit run main.py
   ```

   The browser will open automatically at `http://localhost:8501`.

---

## 📋 Usage guide

| Step | Action                                                                                                                                          |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | In **Huấn luyện & Dự đoán**, paste an Amazon product URL *or* upload a custom CSV containing at least a **rating** & **comment** column.        |
| 2    | Select the appropriate columns, click **Gộp dữ liệu** to merge.                                                                                 |
| 3    | Choose cleaning operations and press **Làm sạch**.                                                                                              |
| 4    | (Optional) Augment the first *N* comments with extra synthetic rows.                                                                            |
| 5    | Pick vectorisers, click **Vector hóa văn bản**.                                                                                                 |
| 6    | Select label column & test‑split, hit **Huấn luyện tất cả mô hình**. Review metrics & confusion matrices; the highest‑accuracy model is stored. |
| 7    | Enter any sentence in the **Dự đoán** box to see the predicted label.                                                                           |
| 8    | Switch to **🤖 chatbot** for open‑domain Q\&A, or to **🎬 Đề xuất phim** to rate titles and get recommendations.                                |

---

## 🧪 Datasets & default artefacts

| File                         | Source                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **amazon\_reviews.csv**      | Auto‑generated by the crawler (one row per review).                                   |
| **movies.csv / ratings.csv** | [MovieLens 1M](https://grouplens.org/datasets/movielens/) – packaged for convenience. |

---

## 📦 Dependencies (excerpt)

* **Python ≥ 3.9**
* Streamlit, pandas, numpy
* scikit‑learn, seaborn, matplotlib
* spaCy + *en\_core\_web\_sm*, gensim
* openai (for ChatGPT)
* requests, BeautifulSoup4

See `requirements.txt` for the full list & exact versions.

---

## 🤝 Contributing

1. Fork the repo and create your branch: `git checkout -b feature/my‑feature`.
2. Commit your changes: `git commit -m 'Add awesome feature'`.
3. Push to the branch: `git push origin feature/my‑feature`.
4. Open a pull request.

Bug reports and feature requests are also welcome via Issues.

---

## 🛡️ License

This project is released under the **MIT License** – see `LICENSE` for details.

---

## 💬 Acknowledgements

* [OpenAI](https://openai.com) for the GPT‑4o API
* [MovieLens](https://grouplens.org) for the ratings dataset
* All open‑source package authors whose tools power this demo.
