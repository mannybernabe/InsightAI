# InsightAI 🤖

An advanced AI-powered search and conversation platform that delivers intelligent, context-aware interactions with transparent reasoning processes.

## Features

- 🧠 **Intelligent Reasoning Display**: Watch the AI's thought process as it analyzes your questions
- 🔍 **Smart Web Search Integration**: 
  - Automatically determines when to use news vs general search based on query context
  - Real-time web search results displayed alongside AI responses
  - Intelligent topic detection for optimized search results
- 💬 **Context-Aware Conversations**: Maintains conversation history for more coherent interactions
- 🌐 **Dual Search Modes**:
  - News Search: Optimized for recent events, sports, and current affairs
  - General Search: Best for historical information, concepts, and how-to queries

## Technology Stack

- **Frontend**: Streamlit
- **AI Integration**: Groq API
- **Web Search**: Tavily API
- **Language**: Python 3.11

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|-----------|
| GROQ_API_KEY | API key for Groq's AI services | Yes |
| TAVILY_API_KEY | API key for Tavily's search services | Yes |

## Usage

1. Start the application:
```bash
streamlit run streamlit_app.py
```

2. Access the web interface at `http://localhost:5000`

3. Enable web search using the toggle at the top of the interface

4. Start asking questions! The AI will:
   - Analyze your query to determine the best search approach
   - Display its reasoning process
   - Show relevant search results
   - Provide a comprehensive response

## Search Functionality

The application uses an intelligent search system that automatically determines whether to use "news" or "general" search based on your query:

### News Search
- Recent events
- Sports games
- Current affairs
- Breaking news
- Market updates

### General Search
- Historical information
- Conceptual questions
- How-to queries
- General knowledge

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
