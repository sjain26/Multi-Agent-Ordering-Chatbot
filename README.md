# Multi-Agent Ordering Chatbot 🤖

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/jainsatyam26/intent-identifier)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An intelligent multi-agent chatbot system that automatically classifies and routes customer orders to specialized agents for efficient order processing.

🔗 **Live Demo**: [https://huggingface.co/spaces/jainsatyam26/intent-identifier](https://huggingface.co/spaces/jainsatyam26/intent-identifier)

## 🌟 Features

### Core Functionality
- **🤖 Multi-Agent System**: Three specialized agents working together
  - **Orchestrator Agent**: Initial greeting and intelligent order classification
  - **Generic Order Agent**: Handles standard/personal orders (< 50 units)
  - **Bulk Order Agent**: Processes large quantity/wholesale orders (≥ 50 units)

### Advanced Features
- **🧠 AI-Powered Classification**: Uses Groq AI (Mixtral model) for intelligent order routing
- **💾 Persistent Storage**: SQLite database for order and conversation history
- **🔒 Security Features**:
  - Input validation and sanitization
  - SQL injection prevention
  - Rate limiting (10 requests/minute)
  - Session isolation
- **📊 Comprehensive Logging**: File and console logging with multiple log levels
- **⚡ Performance Optimization**:
  - 30-second timeout protection
  - Efficient session management
  - Error recovery mechanisms
- **🎨 Professional UI**: Clean Gradio interface with real-time chat

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Groq API key (get it from [console.groq.com/keys](https://console.groq.com/keys))

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone https://github.com/sjain26/Multi-Agent-Ordering-Chatbot.git
   cd Multi-Agent-Ordering-Chatbot
   \`\`\`

2. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Set up environment variables**
   \`\`\`bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your Groq API key
   # Or export directly:
   export GROQ_API_KEY="your_groq_api_key_here"
   \`\`\`

4. **Run the application**
   \`\`\`bash
   python app.py
   \`\`\`

5. **Access the chatbot**
   - Local: http://localhost:7860
   - Network: http://[your-ip]:7860

## 📖 How to Use

1. **Start a conversation**: The chatbot will greet you and ask for an order title
2. **Provide order details**: Describe what you want to order
3. **Automatic routing**: The system classifies your order and routes to the appropriate agent
4. **Follow the prompts**: Answer the agent's questions about:
   - Product name/type
   - Quantity needed
   - Brand/supplier preferences
5. **Review your order**: Get a formatted summary with JSON output
6. **Place another order**: The system resets for your next order

## 🏗️ Architecture

### System Components
\`\`\`
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         │
┌────────▼────────┐
│  Orchestrator   │ ◄── Initial classification
│     Agent       │     and routing
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│Generic│ │ Bulk  │ ◄── Specialized order
│ Agent │ │ Agent │     processing
└───┬───┘ └──┬────┘
    │        │
┌───▼────────▼───┐
│   Database     │ ◄── Order storage
│   (SQLite)     │     and logging
└────────────────┘
\`\`\`

### Technology Stack
- **Backend**: Python with LangChain framework
- **AI Model**: Groq API (Mixtral-8x7b-32768)
- **Frontend**: Gradio for web interface
- **Database**: SQLite for data persistence
- **Logging**: Python logging module

## 🔧 Configuration

### Environment Variables
- \`GROQ_API_KEY\`: Your Groq API key (required)
- \`GRADIO_SERVER_PORT\`: Custom port (optional, default: 7860)

### Customizable Parameters (in app.py)
\`\`\`python
MAX_INPUT_LENGTH = 500      # Maximum characters per input
MAX_TITLE_LENGTH = 100      # Maximum order title length
MAX_DESCRIPTION_LENGTH = 1000  # Maximum description length
API_TIMEOUT = 30            # API call timeout in seconds
RATE_LIMIT_CALLS = 10       # Max API calls per window
RATE_LIMIT_WINDOW = 60      # Rate limit time window in seconds
\`\`\`

## 📊 Database Schema

### Conversation Table
- \`id\`: Primary key
- \`session_id\`: Unique session identifier
- \`timestamp\`: Message timestamp
- \`user_input\`: User's message
- \`chatbot_response\`: Bot's response
- \`agent\`: Active agent name

### Order Table
- \`id\`: Primary key
- \`session_id\`: Session reference
- \`title\`: Order title
- \`description\`: Detailed description
- \`product_name\`: Product/type
- \`quantity\`: Number of units
- \`brand_preference\`: Preferred brand/supplier
- \`created_at\`: Order timestamp

## 🛡️ Security Features

1. **Input Validation**
   - Empty input prevention
   - Character length limits
   - Special character sanitization

2. **Error Handling**
   - Global exception handling
   - Graceful error recovery
   - User-friendly error messages

3. **Rate Limiting**
   - 10 requests per minute per session
   - Automatic cleanup of old timestamps

4. **Data Protection**
   - Parameterized SQL queries
   - Session isolation
   - No hardcoded credentials

## 📈 Monitoring

- **Log File**: \`chatbot.log\` contains all application events
- **Console Output**: Real-time status updates
- **Database**: Query \`chatbot.db\` for historical data

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (\`git checkout -b feature/AmazingFeature\`)
3. Commit your changes (\`git commit -m 'Add some AmazingFeature'\`)
4. Push to the branch (\`git push origin feature/AmazingFeature\`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) framework
- Powered by [Groq AI](https://groq.com/) for LLM capabilities
- UI created with [Gradio](https://gradio.app/)
- Deployed on [Hugging Face Spaces](https://huggingface.co/spaces)

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues before creating new ones
- Provide detailed information for bug reports

---

Made  for Systango
