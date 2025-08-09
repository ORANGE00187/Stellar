# Stellar ✨ The AI-Powered Creation Engine

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)

> Stellar is not just an AI assistant; it's an integrated development and research environment.  
> Built with a powerful Python (Flask) backend and a dynamic frontend, Stellar leverages multiple large language models to provide a suite of specialized tools that go far beyond simple chat.

<!-- Add a high-quality GIF of the UI in action! -->
<!-- ![Stellar UI Demo](link-to-your-demo.gif) -->

---

## 🌟 Core Features: The Five Modes of Creation

### **Stellar Mode**
The foundational conversational AI experience.

- **Dynamic Model Selection**: Switch between AI models (Emerald, Lunarity, Crimson, Obsidian).
- **Full Chat History**: All conversations are saved and searchable.

---

### 🔬 **Spectrum Mode**: The AI Research Analyst

Transform a simple query into a comprehensive research paper.

- **Deep Web Research**: Powered by Tavily API's "Spectral Search".
- **Content Synthesis**: Scrapes and analyzes real-time web data.
- **Tangible Output**: Generates detailed Markdown/HTML papers with citations and solution proposals.

---

### 🚀 **Nebula Mode**: The AI Application Architect

Build full-stack web applications from a single prompt.

**4-Step Guided Process**:
1. **Plan**: Blueprint for frontend and backend logic.
2. **Frontend**: HTML, CSS, JS code generation.
3. **Backend**: Flask-based Python server.
4. **Verify**: Static analysis for code integrity.

- **Instant Deployment**: Run in a built-in sandbox with live previews via ngrok.

---

### 📊 **Cosmos Mode**: The AI Data Artist

Turn raw data into stunning, interactive reports.

- **Data Analysis**: From files (CSV, etc.) or scraped content.
- **Beautiful Output**: Generates Tailwind CSS reports with Chart.js visualizations.

---

### 💻 **CodeLabs Mode**: The AI Coding Dojo

Practice coding skills with real-time AI mentorship.

- **Dynamic Problem Generation**: Based on topic and difficulty.
- **AI Mentor Panel**: Hints, code reviews, complexity analysis.
- **Live IDE**: Write, run, and test code in a secure sandbox.
- **Interview Simulation**: Follow-up Q&A to test your understanding.

---

## 🧠 Key Platform Features

- **Secure Code Execution**: Supports Python, JS, C++, Java, etc. inside Docker sandboxes.
- **Ngrok Integration**: Auto-generated live web previews for Flask apps.
- **Advanced File Analysis**: Supports PDFs, DOCX, images, code files, etc.
- **Persistent Chat & User Management**: Cloud-synced with SQLite Cloud.
- **Dynamic UI Themes**: UI adapts based on selected AI model.

---

## 🛠️ Tech Stack

### 🔧 Backend
- **Framework**: Flask
- **Database**: SQLite Cloud

### 🧠 AI & Data
- `google-generativeai` (Gemini Models)  
- `tavily-python` (Web Search)  
- `requests`, `BeautifulSoup4` (Web Scraping)  
- `PyPDF2` (PDF Parsing)

### ⚙️ DevOps & Execution
- Docker (Secure Code Sandbox)  
- Ngrok (Live Web Previews)

### 💻 Frontend
- **HTML5, CSS3, Vanilla JS (ES6+)**
- **Server-Sent Events (SSE)**: Real-time updates

**Libraries**:
- `Marked.js`: Markdown to HTML  
- `Highlight.js`: Syntax highlighting  
- `KaTeX`: Math rendering  
- `Turndown.js`: HTML to Markdown  

---

## 🚀 Getting Started

### 📋 Prerequisites
- Python **3.10+**
- Docker & Docker Compose
- Ngrok authtoken

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Stellar.git
cd Stellar


2️⃣ Set Up the Python Environment
pip install -r requirements.txt


3️⃣ Configure Environment Variables
Create a .env file using keys.env.example as a template.
# Essential Keys
SECRET_KEY="a-very-strong-and-random-secret-key"
SQL_API_KEY="your_sqlite_cloud_api_key"
NGROK_AUTHTOKEN="your_ngrok_authtoken"
ADMIN_PASSWORD="your_super_secure_admin_password"

# Google AI API Keys
PRIMARY_API_KEY="your_main_google_ai_api_key"
BACKUP_API_KEY_1="your_backup_google_ai_key_1"

# Feature-Specific API Keys
TAVILY_API_KEY="your_tavily_api_key"
NEBULA_KEY_STEP1="..."


4️⃣ Build the Docker Sandbox Images
python dockersetup.py

This will create Dockerfiles and build environments for:

Python
JavaScript
Java
C++
And more...


5️⃣ Run the Application
flask run --host=0.0.0.0 --port=5013

Access the platform at:
📍 http://localhost:5013

🤝 How to Contribute
We welcome contributions!


Fork the repository.


Create a branch:
git checkout -b feature/YourAmazingFeature



Make changes & commit:
git commit -m 'Add some AmazingFeature'



Push the branch:
git push origin feature/YourAmazingFeature



Open a Pull Request.



📜 License
This project is licensed under the MIT License.
See the LICENSE file for full details.
