# 🎙️ HealthPodcastIQ

**Transforming health podcasts into actionable, verified knowledge.**

HealthPodcastIQ is an AI-driven intelligent system designed to help you get more out of the health podcasts you love. Whether you're a medical student, a fitness enthusiast, or just curious about wellness, we bridge the gap between "just listening" and "actually learning."

---

## ✨ What does it do?

Imagine listening to a 2-hour deep dive on nutrition. Instead of frantically taking notes, HealthPodcastIQ does the heavy lifting for you:

- **📝 Smart Transcripts**: Converts audio to text with high accuracy, even for complex medical terms.
- **📚 Multi-Level Summaries**: Need a 1-minute brief? Or a 5-minute deep dive? We've got you covered with Short, Medium, and Detailed versions.
- **❓ Flashcard-Ready QAs**: Automatically generates 10-15 study pairs to help you test what you learned.
- **✅ Science-Backed Fact-Checking**: We verify claims made in the podcast against scientific sources, flagging what’s proven and what’s disputed.
- **🛣️ Intelligent Learning Paths**: If you have multiple episodes, we’ll sequence them from Beginner to Advanced so you always have a logical next step.
- **🔊 Accessibility for All**: Not in the mood to read? We synthesize everything back into audio summaries you can listen to on the go.

---

## 🚀 Getting Started

### Prerequisites
- **Node.js**: For the sleek frontend.
- **Python 3.10+**: For the heavy AI processing.

### Run Locally

1. **Clone & Install**:
   ```bash
   git clone https://github.com/im-subhamkumar/HealthPodcastIQ.git
   cd HealthPodcastIQ
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Backend Setup**:
   ```bash
   cd ../backend
   pip install -r requirements.txt
   python app.py
   ```

4. **Environment Variables**:
   Check the `.env.example` files in both the root and `frontend/` folders to set up your API keys.

---

## 🛠️ The Tech Behind the IQ

We use a "5-Pillar" AI architecture to ensure quality:
- **Whisper v3**: For industry-leading transcription.
- **BART**: For semantic-aware summarization.
- **Flan-T5**: For pedagogical Q&A generation.
- **SpaCy & DuckDuckGo**: For real-world fact verification.
- **BERT**: For intelligent content sequencing.

---

## 🤝 Contributing

We're building this for the health community! If you have ideas or want to improve the AI models, feel free to open an issue or submit a PR.

---

*Made with ❤️ for better health education.*
