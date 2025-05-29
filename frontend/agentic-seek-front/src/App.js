import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './App.css';
import { colors } from './colors';

function App() {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentView, setCurrentView] = useState('chat');
    const [responseData, setResponseData] = useState(null);
    const [isOnline, setIsOnline] = useState(false);
    const [status, setStatus] = useState('Agents ready');
    const [expandedReasoning, setExpandedReasoning] = useState(new Set());
    const [currentPlan, setCurrentPlan] = useState([]);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        const intervalId = setInterval(() => {
            checkHealth();
            fetchLatestAnswer();
            fetchScreenshot();
            fetchCurrentPlan();
        }, 3000);
        return () => clearInterval(intervalId);
    }, [messages]);

    const checkHealth = async () => {
        try {
            await axios.get('http://127.0.0.1:8000/health');
            setIsOnline(true);
            console.log('System is online');
        } catch {
            setIsOnline(false);
            console.log('System is offline');
        }
    };

    const fetchScreenshot = async () => {
        try {
            const timestamp = new Date().getTime();
            const res = await axios.get(`http://127.0.0.1:8000/screenshots/updated_screen.png?timestamp=${timestamp}`, {
                responseType: 'blob'
            });
            console.log('Screenshot fetched successfully');
            const imageUrl = URL.createObjectURL(res.data);
            setResponseData((prev) => {
                if (prev?.screenshot && prev.screenshot !== 'placeholder.png') {
                    URL.revokeObjectURL(prev.screenshot);
                }
                return {
                    ...prev,
                    screenshot: imageUrl,
                    screenshotTimestamp: new Date().getTime()
                };
            });
        } catch (err) {
            console.error('Error fetching screenshot:', err);
            setResponseData((prev) => ({
                ...prev,
                screenshot: 'placeholder.png',
                screenshotTimestamp: new Date().getTime()
            }));
        }
    };

    const fetchCurrentPlan = async () => {
        try {
            const res = await axios.get('http://127.0.0.1:8000/current_plan');
            if (res.data.plan) {
                setCurrentPlan(res.data.plan);
            }
        } catch (error) {
            console.error('Error fetching current plan:', error);
        }
    };

    const normalizeAnswer = (answer) => {
        return answer
            .trim()
            .toLowerCase()
            .replace(/\s+/g, ' ')
            .replace(/[.,!?]/g, '')
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const toggleReasoning = (messageIndex) => {
        setExpandedReasoning(prev => {
            const newSet = new Set(prev);
            if (newSet.has(messageIndex)) {
                newSet.delete(messageIndex);
            } else {
                newSet.add(messageIndex);
            }
            return newSet;
        });
    };

    const fetchLatestAnswer = async () => {
        try {
            const res = await axios.get('http://127.0.0.1:8000/latest_answer');
            const data = res.data;

            updateData(data);
            if (!data.answer || data.answer.trim() === '') {
                return;
            }
            const normalizedNewAnswer = normalizeAnswer(data.answer);
            const answerExists = messages.some(
                (msg) => normalizeAnswer(msg.content) === normalizedNewAnswer
            );
            if (!answerExists) {
                setMessages((prev) => [
                    ...prev,
                    {
                        type: 'agent',
                        content: data.answer,
                        reasoning: data.reasoning,
                        agentName: data.agent_name,
                        status: data.status,
                        uid: data.uid,
                    },
                ]);
                setStatus(data.status);
                scrollToBottom();
            } else {
                console.log('Duplicate answer detected, skipping:', data.answer);
            }
        } catch (error) {
            console.error('Error fetching latest answer:', error);
        }
    };

    const updateData = (data) => {
        setResponseData((prev) => ({
            ...prev,
            blocks: data.blocks || prev.blocks || null,
            done: data.done,
            answer: data.answer,
            agent_name: data.agent_name,
            status: data.status,
            uid: data.uid,
        }));
    };

    const handleStop = async (e) => {
        e.preventDefault();
        checkHealth();
        setIsLoading(false);
        setError(null);
        try {
            const res = await axios.get('http://127.0.0.1:8000/stop');
            setStatus("Requesting stop...");
        } catch (err) {
            console.error('Error stopping the agent:', err);
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        checkHealth();
        if (!query.trim()) {
            console.log('Empty query');
            return;
        }
        setMessages((prev) => [...prev, { type: 'user', content: query }]);
        setIsLoading(true);
        setError(null);

        try {
            console.log('Sending query:', query);
            setQuery('waiting for response...');
            const res = await axios.post('http://127.0.0.1:8000/query', {
                query,
                tts_enabled: false
            });
            setQuery('Enter your query...');
            console.log('Response:', res.data);
            const data = res.data;
            updateData(data);
        } catch (err) {
            console.error('Error:', err);
            setError('Failed to process query.');
            setMessages((prev) => [
                ...prev,
                { type: 'error', content: 'Error: Unable to get a response.' },
            ]);
        } finally {
            console.log('Query completed');
            setIsLoading(false);
            setQuery('');
        }
    };

    const handleGetScreenshot = async () => {
        try {
            setCurrentView('screenshot');
        } catch (err) {
            setError('Browser not in use');
        }
    };

    const handleShowChat = () => {
        setCurrentView('chat');
    };

    const handleShowPlan = () => {
        setCurrentView('plan');
    };

    return (
        <div className="app" style={{ backgroundColor: '#000000' }}>
            <header className="header">
                <h1>Novah</h1>
            </header>
            <main className="main">
                <div className="app-sections">
                    <div className="chat-section">
                        <h2>Chat Interface</h2>
                        <div className="messages">
                            {messages.length === 0 ? (
                                <p className="placeholder">No messages yet. Type below to start!</p>
                            ) : (
                                messages.map((msg, index) => (
                                    <div
                                        key={index}
                                        className={`message ${
                                            msg.type === 'user'
                                                ? 'user-message'
                                                : msg.type === 'agent'
                                                ? 'agent-message'
                                                : 'error-message'
                                        }`}
                                    >
                                        <div className="message-header">
                                            {msg.type === 'agent' && (
                                                <span className="agent-name">{msg.agentName}</span>
                                            )}
                                            {msg.type === 'agent' && msg.reasoning && expandedReasoning.has(index) && (
                                                <div className="reasoning-content">
                                                    <ReactMarkdown>{msg.reasoning}</ReactMarkdown>
                                                </div>
                                            )}
                                            {msg.type === 'agent' && (
                                                <button 
                                                    className="reasoning-toggle"
                                                    onClick={() => toggleReasoning(index)}
                                                    title={expandedReasoning.has(index) ? "Hide reasoning" : "Show reasoning"}
                                                >
                                                    {expandedReasoning.has(index) ? '▼' : '▶'} Reasoning
                                                </button>
                                            )}
                                        </div>
                                        <div className="message-content" style={{ backgroundColor: 'transparent' }}>
                                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                                        </div>
                                    </div>
                                ))
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                        {isOnline && <div className="loading-animation">{status}</div>}
                        {!isLoading && !isOnline && <p className="loading-animation">System offline. Deploy backend first.</p>}
                        <form onSubmit={handleSubmit} className="input-form">
                            <input
                                type="text"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder={isOnline ? 'Enter your query...' : 'System offline.'}
                                disabled={isLoading || !isOnline}
                            />
                            <button type="submit" disabled={isLoading}>
                                Send
                            </button>
                            <button onClick={handleStop}>
                                Stop
                            </button>
                        </form>
                    </div>

                    <div className="computer-section">
                        <h2>Computer View</h2>
                        <div className="view-selector">
                            <button
                                className={currentView === 'blocks' ? 'active' : ''}
                                onClick={() => setCurrentView('blocks')}
                            >
                                Editor View
                            </button>
                            <button
                                className={currentView === 'screenshot' ? 'active' : ''}
                                onClick={responseData?.screenshot ? () => setCurrentView('screenshot') : handleGetScreenshot}
                            >
                                Browser View
                            </button>
                            <button
                                className={currentView === 'plan' ? 'active' : ''}
                                onClick={() => setCurrentView('plan')}
                            >
                                Plan View
                            </button>
                        </div>
                        <div className="content">
                            {error && <p className="error">{error}</p>}
                            {currentView === 'blocks' ? (
                                <div className="blocks">
                                    {responseData && responseData.blocks && Object.values(responseData.blocks).length > 0 ? (
                                        Object.values(responseData.blocks).map((block, index) => (
                                            <div key={index} className="block">
                                                <p className="block-tool">Tool: {block.tool_type}</p>
                                                <pre>{block.block}</pre>
                                                <p className="block-feedback">Feedback: {block.feedback}</p>
                                                {block.success ? (
                                                    <p className="block-success">Success</p>
                                                ) : (
                                                    <p className="block-failure">Failure</p>
                                                )}
                                            </div>
                                        ))
                                    ) : (
                                        <div className="block">
                                            <p className="block-tool">Tool: No tool in use</p>
                                            <pre>No file opened</pre>
                                        </div>
                                    )}
                                </div>
                            ) : currentView === 'screenshot' ? (
                                <div className="screenshot">
                                    <img
                                        src={responseData?.screenshot || 'placeholder.png'}
                                        alt="Screenshot"
                                        onError={(e) => {
                                            e.target.src = 'placeholder.png';
                                            console.error('Failed to load screenshot');
                                        }}
                                        key={responseData?.screenshotTimestamp || 'default'}
                                    />
                                </div>
                            ) : (
                                <div className="plan-view">
                                    <h2>Current Plan</h2>
                                    {currentPlan && currentPlan.length > 0 ? (
                                        <div className="plan-list">
                                            {currentPlan.map((task, index) => (
                                                <div key={index} className="plan-item">
                                                    <h3>Task {index + 1}: {task.task_name}</h3>
                                                    <p><strong>Agent:</strong> {task.agent}</p>
                                                    <p><strong>Description:</strong> {task.description}</p>
                                                    {task.requirements && (
                                                        <p><strong>Requirements:</strong> {task.requirements}</p>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <p>No active plan</p>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}

export default App;