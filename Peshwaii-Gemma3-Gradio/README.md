# 📜 Peshwai Historian AI – Marathi Chatbot

**An interactive Gradio app to explore lesser-known historical facts from the Peshwa dynasty of Pune — in pure Marathi.**

> 🤖 Powered by a fine-tuned version of [Gemma 3 12B](https://huggingface.co/unsloth/gemma-3-12b-it-unsloth-bnb-4bit) on curated Marathi historical data using [Unsloth](https://github.com/unslothai/unsloth).

---

## 🧠 What Is This?

This chatbot acts like a knowledgeable Peshwai-era historian who can:

- 🧾 Answer historical questions in Marathi
- 🔍 Focus on less-known events, policies, and individuals beyond Shaniwarwada or Bajirao
- 🗣️ Generate elegant, research-style answers using few-shot prompting

---

## ✨ Features

✅ Natural Marathi generation  
✅ Preloaded prompt examples  
✅ Answer download as PDF  
✅ Feedback collection  
✅ CSV logging for analytics  
✅ Clean mobile-friendly Gradio UI  

---

## 🧪 How to Use

1. Type your question in Marathi (e.g. "महादजी शिंदे आणि इंग्रज यांचे संबंध कसे होते?")
2. Click `उत्तर मिळवा 🚀`
3. View a detailed, researched answer
4. Optionally download as PDF or rate the answer

---

## 📁 Files

| File           | Description                              |
|----------------|------------------------------------------|
| `app.py`       | Main Gradio app with UI + logic          |
| `requirements.txt` | Dependencies (`gradio`, `unsloth`, etc.) |
| `answers.csv`  | Logs user questions + model responses    |
| `feedback.csv` | Stores ratings from users                |

---

## 💻 Running Locally (optional)

```bash
git clone https://huggingface.co/spaces/<your-username>/peshwai-marathi-ai
cd peshwai-marathi-ai
pip install -r requirements.txt
python app.py
