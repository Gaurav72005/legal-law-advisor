# ⚖️ Motor & Cyber Law Advisor (RAG-based Legal AI Assistant)

An AI-powered legal assistant designed to provide accurate, context-grounded answers related to:

* **Motor Vehicles Act, 1988 (India)**
* **Information Technology Act, 2000 (India)**

This system leverages a **Retrieval-Augmented Generation (RAG)** pipeline to ensure that all responses are based strictly on retrieved legal text, minimizing hallucinations and improving trustworthiness.

---

## 🚀 Key Features

* ✅ **Grounded Legal Responses** (No hallucinations)
* 📚 **Section-wise Citations** from official Acts
* ⚡ **Low-latency inference** using Groq LLM
* 🔍 **Semantic Search** via ChromaDB
* 💬 **Interactive Chat UI** (Streamlit-based)
* 📊 **Query Logging & Analytics** (SQLite)
* 🔁 **LLM Fallback Mechanism** (Groq → Gemini)

---

## 🧠 System Architecture

```
User Query (Streamlit UI)
        ↓
Retrieval Chain (LangChain)
        ↓
ChromaDB (Top-k Similar Legal Chunks)
        ↓
Prompt Construction (Strict Grounding Rules)
        ↓
LLM (Groq / Gemini)
        ↓
Response + Legal Citations
        ↓
SQLite Logging
```

---

## 🛠️ Tech Stack

| Layer                 | Technology                       |
| --------------------- | -------------------------------- |
| Frontend              | Streamlit                        |
| Backend Orchestration | LangChain 1.x                    |
| Vector Database       | ChromaDB (Local Persistent)      |
| Embeddings            | HuggingFace (`all-MiniLM-L6-v2`) |
| LLM                   | Groq (`llama-3.3-70b-versatile`) |
| Fallback LLM          | Google Gemini 2.5 Pro            |
| Logging               | SQLite                           |

---

## 📁 Project Structure

```
Motor-Cyber-Law-Advisor/
│
├── webscraper.py        # Extracts legal text from official PDFs
├── ingestion.py         # Cleans, chunks, embeds & stores in ChromaDB
├── retrievalchain.py    # Core RAG pipeline + LLM + logging
├── streamlitui.py       # Chat UI frontend
│
├── data/                # ChromaDB persistence directory
├── logs.db              # SQLite database
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/motor-cyber-law-advisor.git
cd motor-cyber-law-advisor
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Set Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_gemini_api_key
```

---

### 4. Run Data Pipeline

#### Step 1: Scrape Legal Documents

```bash
python webscraper.py
```

#### Step 2: Generate Embeddings & Store in ChromaDB

```bash
python ingestion.py
```

---

### 5. Start the Application

```bash
streamlit run streamlitui.py
```

---

## 🔍 How It Works

### 1. Retrieval Phase

* User query is embedded
* ChromaDB performs similarity search (`k=3`)
* Top legal chunks are retrieved

### 2. Augmentation Phase

* Retrieved context is injected into a strict prompt template
* Ensures the model **only answers from provided legal text**

### 3. Generation Phase

* Query sent to Groq LLM
* If failure occurs → fallback to Gemini

### 4. Logging

* Query, response, latency stored in SQLite

---

## 📊 Example Query

> "What is the penalty for driving without a license in India?"

**Response:**

* Based strictly on Motor Vehicles Act
* Includes section citation (e.g., Section 3/181)
* Adds disclaimer if context is insufficient

---

## 🧩 Future Enhancements

* 🔄 Hybrid Search (Semantic + BM25)
* 🧠 Cross-Encoder Reranking
* 📈 Confidence Scoring
* 🗂 Section-aware Chunking
* 🌐 Cloud Deployment (AWS / GCP)
* 📊 Advanced Analytics Dashboard
* 💬 Multi-turn Memory Support

---

## ⚠️ Disclaimer

This system is intended for **educational and informational purposes only**.
It does **not constitute legal advice**. Always consult a qualified legal professional for official guidance.

---

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed as part of a legal-tech RAG system using modern LLM infrastructure.

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!
