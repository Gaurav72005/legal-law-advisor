import { useState, useRef, useEffect } from 'react';
import './index.css';

const API_BaseURL = "http://127.0.0.1:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [inputVal, setInputVal] = useState('');
  const [loading, setLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState('Checking...');
  const chatEndRef = useRef(null);

  useEffect(() => {
    // Scroll to bottom whenever messages change
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    // Check server status
    fetch(`${API_BaseURL}/api/status`)
      .then(res => res.json())
      .then(data => setServerStatus(`⚡ ${data.provider === 'groq' ? 'Groq · Llama 3.3' : 'Gemini 2.5 Pro'}`))
      .catch(() => setServerStatus("Offline"));
  }, []);

  const suggestions = [
    "Fine for drunk driving?",
    "Penalty for no driving licence?",
    "What is Section 66C IT Act?",
    "Is hacking a crime in India?",
    "Can police seize my vehicle?",
    "What is cyber terrorism?",
  ];

  const handleSend = async (queryText) => {
    if (!queryText.trim()) return;

    const newMsg = { id: Date.now(), role: 'user', content: queryText };
    setMessages(prev => [...prev, newMsg]);
    setInputVal('');
    setLoading(true);

    try {
      const resp = await fetch(`${API_BaseURL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText })
      });

      if (!resp.ok) {
        throw new Error("Server error " + resp.status);
      }
      const data = await resp.json();

      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: data.answer,
        chunks: data.chunks,
        latency_ms: data.latency_ms,
        answer_found: data.answer_found
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: "Error connecting to legal database: " + err.message,
        error: true
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([]);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-logo">⚖</div>
        <div>
          <h1>Motor & Cyber Law Advisor</h1>
          <div className="badge-container" style={{ marginTop: '4px' }}>
            <span className="badge badge-green">Motor Vehicles Act · IT Act 2000</span>
            <span className="badge badge-purple">{serverStatus}</span>
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <div className="chat-container">
        {messages.length === 0 ? (
          <div className="suggestions-wrapper">
            <h2 className="suggestions-title">Try asking one of these:</h2>
            <div className="suggestions-grid">
              {suggestions.map((s, i) => (
                <button 
                  key={i} 
                  className="suggestion-btn"
                  onClick={() => handleSend(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map(msg => (
              <div key={msg.id} className={`message-wrapper ${msg.role} ${msg.answer_found === 0 || msg.error ? 'disclaimer' : ''}`}>
                <div className="bubble">
                  {msg.content}
                </div>
                {msg.chunks && msg.chunks.length > 0 && (
                  <div className="source-line">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                    </svg>
                    {[...new Set(msg.chunks.map(c => c.metadata.act_name || c.metadata.act_code))].map((sourceName, idx) => (
                      <span key={idx} className="badge badge-green">{sourceName}</span>
                    ))}
                    <span style={{opacity: 0.7}}>· {msg.latency_ms} ms</span>
                  </div>
                )}
              </div>
            ))}
            
            {loading && (
              <div className="message-wrapper bot">
                <div className="bubble">
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}
            
            {messages.length > 0 && !loading && (
              <button className="clear-btn" onClick={handleClear}>
                🗑 Clear chat
              </button>
            )}
          </>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-area">
        <div className="input-wrapper">
          <input 
            type="text" 
            className="msg-input" 
            placeholder="Ask a legal question... (e.g. Fine for drunk driving?)" 
            value={inputVal}
            onChange={e => setInputVal(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') handleSend(inputVal);
            }}
            disabled={loading}
          />
        </div>
        <button 
          className="send-btn" 
          onClick={() => handleSend(inputVal)}
          disabled={!inputVal.trim() || loading}
          title="Send message"
        >
          <svg className="send-icon" viewBox="0 0 24 24">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
          </svg>
        </button>
      </div>
    </div>
  );
}

export default App;
