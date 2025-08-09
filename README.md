Stellar ✨ The AI-Powered Creation Engine

![alt text](https://img.shields.io/badge/License-MIT-blue.svg)


![alt text](https://img.shields.io/badge/python-3.12-blue.svg)


![alt text](https://img.shields.io/badge/Framework-Flask-black.svg)

<!-- Add a high-quality GIF of the UI in action! -->

<!-- ![Stellar UI Demo](link-to-your-demo.gif) -->


Stellar is not just an AI assistant; it's an integrated development and research environment. Built with a powerful Python (Flask) backend and a dynamic frontend, Stellar leverages multiple large language models to provide a suite of specialized tools that go far beyond simple chat.

Core Features: The Five Modes of Creation

Stellar's power is divided into five distinct, high-level modes, each designed to solve complex, real-world problems.

🌌 Stellar Mode

The foundational conversational AI experience. Perfect for quick questions, brainstorming, and general-purpose assistance.

Dynamic Model Selection: Switch between different AI models (Emerald, Lunarity, Crimson, Obsidian) to balance speed and reasoning power.

Full Chat History: All conversations are saved and searchable.

🔬 Spectrum Mode: The AI Research Analyst

Transform a simple query into a comprehensive research paper.

Deep Web Research: Utilizes the Tavily API for its "Spectral Search" to gather real-time, relevant data.

Content Synthesis: Scrapes and analyzes web sources, integrating their content into a cohesive document.

Tangible Output: Generates detailed, well-structured research papers in Markdown and HTML, complete with citations and a novel solution proposal.

🚀 Nebula Mode: The AI Application Architect

Build and design full-stack web applications from a single prompt.

4-Step Guided Process:

Plan: Generates a detailed blueprint for frontend and backend logic.

Frontend: Writes the complete HTML, CSS, and JavaScript code.

Backend: Produces the Python/Flask server code.

Verify: Performs a static analysis of the generated code to ensure cohesion and correctness.

Instant Deployment: The generated backend code is ready to run in the built-in sandbox.

📊 Cosmos Mode: The AI Data Artist

Turn raw data into beautiful, insightful, and interactive reports.

Data-Driven Visuals: Analyzes uploaded files (like CSVs) and web data to find key insights.

Stunning Reports: Generates single-page HTML reports using Tailwind CSS for styling and Chart.js for rich, interactive infographics.

💻 CodeLabs Mode: The AI Coding Dojo

An interactive environment for practicing and mastering coding skills.

Dynamic Problem Generation: The AI creates unique coding challenges on-demand based on topic and difficulty.

AI Mentor Panel: Provides Socratic hints, on-demand code reviews for correctness and efficiency, and explains complex concepts.

Integrated IDE Panel: Write your code in a live editor and run it against AI-generated test cases directly within the secure sandbox.

Interview Simulation: Engages with follow-up questions to test your understanding of time/space complexity and edge cases.

Key Platform Features

Secure Code Execution Sandbox: Run code in any supported language (Python, JS, C++, Java, etc.) in an isolated Docker environment. For web servers, it automatically uses ngrok to generate a live, shareable preview URL.

Advanced File Analysis: Upload and analyze a wide variety of file types (pdf, docx, images, code files). For large documents, Stellar intelligently splits them into chunks for complete analysis.

Full Chat & User Management:

Secure user authentication with hashed passwords.

Persistent, cloud-synced chat history via SQLite Cloud.

Powerful cross-chat search to find any message and jump directly to it.

Dynamic UI & Theming: The entire UI theme dynamically changes to match the aesthetic of the selected AI model.

🛠️ Tech Stack
Backend

Framework: Flask

Database: SQLite Cloud

AI & Data:

google-generativeai (for Gemini Models)

tavily-python (for Web Search)

requests & BeautifulSoup4 (for Web Scraping)

PyPDF2 (for PDF processing)

DevOps & Execution:

Docker (for Secure Code Sandboxing)

Ngrok (for Live Web Previews)

Frontend

Core: HTML5, CSS3, Vanilla JavaScript (ES6+)

Real-time: Server-Sent Events (SSE) for progress updates.

Rendering & UI:

Marked.js (Markdown to HTML)

Highlight.js (Syntax Highlighting)

KaTeX (Math Equations)

Turndown.js (HTML to Markdown)

🚀 Getting Started

Follow these steps to set up and run your own instance of Stellar.

Prerequisites

Python 3.10+

Docker and Docker Compose installed and running.

An ngrok authtoken for the live preview feature.

1. Clone the Repository
code
Bash
download
content_copy
expand_less

git clone https://github.com/your-username/Stellar.git
cd Stellar
2. Set Up the Python Environment
code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
pip install -r requirements.txt
3. Configure Environment Variables

Create a file named .env in the project root and populate it with your API keys and secrets. Use the keys.env.example file as a template.

code
Env
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# Essential Keys
SECRET_KEY="a-very-strong-and-random-secret-key"
SQL_API_KEY="your_sqlite_cloud_api_key"
NGROK_AUTHTOKEN="your_ngrok_authtoken"
ADMIN_PASSWORD="your_super_secure_admin_password"

# Google AI API Keys (at least one is required)
PRIMARY_API_KEY="your_main_google_ai_api_key"
BACKUP_API_KEY_1="your_backup_google_ai_key_1"
# ... add more backup keys if needed

# Feature-Specific API Keys
TAVILY_API_KEY="your_tavily_api_key"
# ... add keys for Nebula, Cosmos, etc.
NEBULA_KEY_STEP1="..."
4. Build the Docker Sandbox Images

The project includes a setup script to automatically build all the necessary Docker images for the CodeLabs and code execution features.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
python dockersetup.py

This script will create and build Dockerfiles for Python, JavaScript, Java, C++, and more.

5. Run the Application
code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
flask run --host=0.0.0.0 --port=5013

You can now access Stellar in your browser at http://localhost:5013.

🤝 How to Contribute

Contributions are welcome! If you'd like to help improve Stellar, please follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/YourAmazingFeature).

Make your changes and commit them (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/YourAmazingFeature).

Open a Pull Request.

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
