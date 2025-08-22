# MindMate – Mental Health Chatbot

💬 **MindMate** is an educational mental health chatbot built with **Streamlit** and **GPT-Neo**. It provides empathetic responses, mood tracking, and context-aware coping exercise suggestions. This project is for **learning and demonstration purposes only** and is **not a medical device**.

---

## 🌟 Features

- **Interactive Chatbot**: Users can share their thoughts and receive empathetic responses powered by GPT-Neo.
- **Context-aware Coping Suggestions**: Automatically suggests relevant exercises like:
  - 4–7–8 Breathing  
  - Box Breathing  
  - Grounding 5–4–3–2–1  
  - Journaling Prompts
- **Sentiment Analysis**: Uses NLTK’s VADER to gauge emotional tone and guide suggestions.
- **Daily Mood Tracking**: Users can log their mood with notes and view trends in a line chart.
- **Mood Log Export**: Download mood history in JSON format for personal tracking.
- **Educational Demo**: Helps learn about LLM integration, Streamlit, and sentiment-based guidance.

---

## 🛠 Tech Stack

- **Python 3.12+**
- **Streamlit** – for interactive web interface  
- **Transformers (Hugging Face)** – GPT-Neo-125M for chatbot responses  
- **NLTK (VADER)** – sentiment analysis  
- **Datetime & JSON** – mood tracking and data storage  

---

🚀 Getting Started
1. Clone the repository:
git clone https://github.com/Rohithasiri/mindmate-chatbot.git
cd mindmate-chatbot
2. Create a virtual environment:
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
3. Install dependencies:
pip install -r requirements.txt
Example requirements.txt:
streamlit
transformers
torch
nltk
4. Run the app:
streamlit run app.py

⚠️ Disclaimer:
This project is educational and not a medical device. It does not provide professional mental health advice.
If you are in crisis or thinking about harming yourself or others, contact local emergency services immediately.

📂 Repository Structure:
mindmate-chatbot/
│
├─ app.py                 # Main Streamlit app
├─ requirements.txt       # Python dependencies
├─ .gitignore             # Ignored files
├─ LICENSE                # MIT License
└─ README.md              # Project documentation

📝 License:
This project is licensed under the MIT License. See LICENSE
 for details.

🙏 Acknowledgements:
Streamlit – for the interactive interface
Hugging Face Transformers – GPT-Neo integration
NLTK – for sentiment analysis
