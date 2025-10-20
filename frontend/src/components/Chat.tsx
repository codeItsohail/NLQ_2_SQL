import React, { useState, useEffect } from "react";
import axios from "axios";

interface Conversation {
  session_id: string;
  messages: Array<{
    id: number;
    ts: string;
    role: string;
    content: string;
    sql_query?: string;
  }>;
}

interface QueryResult {
  answer: string;
  sql: string;
  data: {
    columns: string[];
    rows: any[];
  };
  session_id?: string;
}

interface Agent {
  id: string;
  name: string;
  description?: string;
  db_url: string;
  trained_at?: string;
}

const Chat: React.FC = () => {
  const USER_ID = "demo-user-123";
  
  // App state flow
  const [appState, setAppState] = useState<"initial" | "agent-selection" | "new-connection" | "connected">("initial");
  
  // Database connection states
  const [dbType, setDbType] = useState("postgresql");
  const [host, setHost] = useState("");
  const [dbName, setDbName] = useState("");
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [port, setPort] = useState("5432");
  const [connected, setConnected] = useState(false);
  const [connectionMsg, setConnectionMsg] = useState("");
  const [dbUrl, setDbUrl] = useState("");

  // Agent states
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [agentName, setAgentName] = useState("");
  const [agentDesc, setAgentDesc] = useState("");
  const [showSaveAgent, setShowSaveAgent] = useState(false);

  // Training states
  const [trainRunId, setTrainRunId] = useState<string | null>(null);
  const [trainProgress, setTrainProgress] = useState<number>(0);
  const [trainStatus, setTrainStatus] = useState<string>("");
  const [trainMsg, setTrainMsg] = useState<string>("");

  // Chat states
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // History states
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationIndex, setCurrentConversationIndex] = useState<number>(-1);

  // Editing/UI states
  const [isQuestionEditable, setIsQuestionEditable] = useState(true);
  const [isEditingSQL, setIsEditingSQL] = useState(false);
  const [editedSQL, setEditedSQL] = useState("");

  // Auto-update training progress
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;
    
    if (trainRunId && (trainStatus === "queued" || trainStatus === "running")) {
      intervalId = setInterval(async () => {
        try {
          const response = await fetch(`http://localhost:5000/api/train/status/${trainRunId}`);
          const data = await response.json();
          
          if (data && data.progress !== undefined) {
            setTrainProgress(data.progress);
            setTrainStatus(data.status);
            setTrainMsg(data.message || "");
            
            if (data.status === "succeeded" || data.status === "failed") {
              clearInterval(intervalId!);
            }
          }
        } catch (error) {
          console.error("Error checking training status:", error);
        }
      }, 1000);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [trainRunId, trainStatus]);

  // Load conversation history when connected
  useEffect(() => {
    if (connected && selectedAgent) {
      loadConversationHistory();
    }
  }, [connected, selectedAgent]);

  // Keep edited SQL in sync with latest response
  useEffect(() => {
    setEditedSQL(response?.sql || "");
  }, [response]);

  const loadConversationHistory = async () => {
    if (!selectedAgent) return;
    
    try {
      const response = await fetch(`http://localhost:5000/api/history/${selectedAgent.id}?limit=100`);
      const data = await response.json();
      setConversations(data.conversations || []);
      setCurrentConversationIndex(-1);
    } catch (error) {
      console.error("Error loading conversation history:", error);
    }
  };

  const fetchAgents = async () => {
    try {
      const res = await fetch(`http://localhost:5000/api/agents/${USER_ID}`);
      const data = await res.json();

      // Ensure we always set an array to avoid "agents.map is not a function"
      const list = Array.isArray(data)
        ? data
        : Array.isArray((data as any)?.rows)
          ? (data as any).rows
          : [];

      if (!Array.isArray(list)) {
        console.warn("Unexpected /api/agents response:", data);
      }

      setAgents(list as Agent[]);
      setAppState("agent-selection");
    } catch (error) {
      console.error("Error fetching agents:", error);
      // Fallback to empty list and still open the selection screen
      setAgents([]);
      setAppState("agent-selection");
    }
  };

  const handleTestConnection = async () => {
    try {
      const body = {
        db_type: dbType,
        host,
        db_name: dbName,
        user,
        password,
        port: parseInt(port, 10),
      };

      const res = await fetch("http://localhost:5000/api/test-connection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json();
      if (data.success) {
        setConnected(true);
        setConnectionMsg("✅ " + data.message);
        const built = `${dbType}://${user}:${password}@${host}:${port}/${dbName}`;
        setDbUrl(built);
        setShowSaveAgent(true);
      } else {
        setConnected(false);
        setConnectionMsg("❌ " + data.error);
      }
    } catch (err: any) {
      setConnected(false);
      setConnectionMsg("❌ " + (err.message || "Connection failed"));
    }
  };

  const handleSaveAgent = async () => {
    try {
      const res = await fetch("http://localhost:5000/api/agents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: USER_ID,
          name: agentName,
          description: agentDesc,
          db_url: dbUrl,
          dbType: dbType,
          host: host,
          dbName: dbName,
          user: user,
          password: password,
          port: port,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setSelectedAgent({
          id: data.agent_id,
          name: agentName,
          description: agentDesc,
          db_url: dbUrl
        });
        setShowSaveAgent(false);
        setAppState("connected");
      } else {
        alert("❌ Failed: " + data.error);
      }
    } catch (err: any) {
      alert("❌ Error: " + (err.message || "Save failed"));
    }
  };

  const handleSelectAgent = async (agent: Agent) => {
    try {
      const parsed = parseDbUrl(agent.db_url);
      if (!parsed) {
        alert("Could not parse database URL");
        return;
      }

      const testRes = await fetch("http://localhost:5000/api/test-connection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          db_type: parsed.dbType,
          host: parsed.host,
          db_name: parsed.dbName,
          user: parsed.user,
          password: parsed.password,
          port: parseInt(parsed.port || "5432", 10),
        }),
      });

      const testData = await testRes.json();
      if (testData.success) {
        setConnected(true);
        setSelectedAgent(agent);
        setAppState("connected");
        setConnectionMsg("✅ Connected to " + agent.name);
      } else {
        alert("❌ Connection failed: " + testData.error);
      }
    } catch (error) {
      alert("❌ Error connecting to agent");
    }
  };

  const handleTrainAgent = async () => {
    if (!selectedAgent) return;
    
    try {
      const resp = await fetch("http://localhost:5000/api/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_id: selectedAgent.id }),
      });
      const data = await resp.json();
      if (data.run_id) {
        setTrainRunId(data.run_id);
        setTrainProgress(0);
        setTrainStatus("queued");
        setTrainMsg("Training started...");
      } else {
        alert("Training failed to start: " + (data.error || "Unknown error"));
      }
    } catch (error) {
      alert("Failed to start training: " + error);
    }
  };

  const handleAsk = async () => {
    if (!question.trim() || !selectedAgent) return;
    
    setLoading(true);
    setFeedback(null);
    
    try {
      const res = await axios.post("http://localhost:5000/api/ask", {
        question,
        agent_id: selectedAgent.id,
        execute: true,
        user_id: USER_ID,
        limit: 50,
        session_id: sessionId,
      });
      
      const result: QueryResult = {
        answer: res.data.answer,
        sql: res.data.sql,
        data: res.data.data || { columns: [], rows: [] },
        session_id: res.data.session_id
      };
      
      setResponse(result);
      if (result.session_id) setSessionId(result.session_id);
      setEditedSQL(result.sql || "");
      setCurrentConversationIndex(-1);
      
      setTimeout(() => loadConversationHistory(), 500);
    } catch (err) {
      console.error("Error:", err);
      setResponse({
        answer: "⚠️ Error contacting AI agent",
        sql: "",
        data: { columns: [], rows: [] }
      });
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (valid: boolean) => {
    if (!response || !selectedAgent) return;
    
    try {
      await axios.post("http://localhost:5000/api/feedback", {
        question: currentConversationIndex === -1 
          ? (conversations[conversations.length - 1]?.messages[0]?.content || "")
          : conversations[currentConversationIndex]?.messages[0]?.content || "",
        sql: response.sql,
        valid,
        agent_id: selectedAgent.id,
        user_id: USER_ID,
        answer: response.answer,
      });
      
      setFeedback(valid ? "✅ Marked as correct" : "❌ Marked as incorrect");
      setTimeout(() => setFeedback(null), 3000);
    } catch (err) {
      setFeedback("⚠️ Error sending feedback");
    }
  };

  const handleReset = () => {
    setQuestion("");
    setResponse(null);
    setFeedback(null);
    setCurrentConversationIndex(-1);
    setSessionId(null);
    setEditedSQL("");
    setIsEditingSQL(false);
    setIsQuestionEditable(true);
  };

  // Execute the edited SQL on backend (no LLM), update results panel
  const applyEditedSQL = async () => {
    if (!selectedAgent || !editedSQL.trim()) {
      setIsEditingSQL(false);
      return;
    }
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/execute", {
        agent_id: selectedAgent.id,
        sql: editedSQL,
        limit: 50,
        user_id: USER_ID
      });

      const ok = res.data?.success !== false; // treat undefined as success
      if (ok) {
        const columns = res.data?.columns || [];
        const rows = res.data?.rows || [];
        const sqlText = res.data?.sql ?? editedSQL;
        setResponse(prev => ({
          ...(prev || { answer: "", sql: "", data: { columns: [], rows: [] } }),
          sql: sqlText,
          data: { columns, rows },
          answer: "Query executed successfully."
        }));
      } else {
        setResponse(prev => ({
          ...(prev || { answer: "", sql: "", data: { columns: [], rows: [] } }),
          sql: editedSQL,
          data: { columns: [], rows: [] },
          answer: `Execution error: ${res.data?.error || "Unknown error"}`
        }));
      }
    } catch (e: any) {
      setResponse(prev => ({
        ...(prev || { answer: "", sql: "", data: { columns: [], rows: [] } }),
        sql: editedSQL,
        data: { columns: [], rows: [] },
        answer: `Execution error: ${e?.message || String(e)}`
      }));
    } finally {
      setLoading(false);
      setIsEditingSQL(false);
    }
  };

  const handleDisconnect = () => {
    setConnected(false);
    setSelectedAgent(null);
    setAppState("initial");
    setConnectionMsg("");
    setResponse(null);
    setConversations([]);
    setSessionId(null);
    setTrainRunId(null);
    setTrainStatus("");
  };

  const navigateToPrevious = () => {
    if (currentConversationIndex < conversations.length - 1) {
      const newIndex = currentConversationIndex + 1;
      setCurrentConversationIndex(newIndex);
      loadConversationData(newIndex);
    }
  };

  const navigateToNext = () => {
    if (currentConversationIndex > 0) {
      const newIndex = currentConversationIndex - 1;
      setCurrentConversationIndex(newIndex);
      loadConversationData(newIndex);
    } else if (currentConversationIndex === 0) {
      setCurrentConversationIndex(-1);
      setResponse(null);
    }
  };

  const loadConversationData = (index: number) => {
    if (index >= 0 && index < conversations.length) {
      const conversation = conversations[index];
      const userMessage = conversation.messages.find(m => m.role === "user");
      const assistantMessage = conversation.messages.find(m => m.role === "assistant");
      
      if (userMessage && assistantMessage) {
        setResponse({
          answer: assistantMessage.content,
          sql: assistantMessage.sql_query || "",
          data: { columns: [], rows: [] }
        });
      }
    }
  };

  function parseDbUrl(dbUrl: string) {
    try {
      const normalized = dbUrl.replace(/^postgres:\/\//, "postgresql://");
      const u = new URL(normalized);
      return {
        dbType: u.protocol.replace(":", ""),
        user: decodeURIComponent(u.username || ""),
        password: decodeURIComponent(u.password || ""),
        host: u.hostname || "",
        port: u.port || "",
        dbName: u.pathname?.replace(/^\//, "") || ""
      };
    } catch (e) {
      return null;
    }
  }

  // Render initial screen
  if (appState === "initial") {
    return (
      <div style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px"
      }}>
        <div style={{
          maxWidth: "500px",
          width: "100%",
          background: "white",
          borderRadius: "16px",
          boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
          padding: "40px"
        }}>
          <h1 style={{ textAlign: "center", color: "#333", marginBottom: "10px", fontSize: "28px" }}>
            🧠 AI SQL Chat
          </h1>
          <p style={{ textAlign: "center", color: "#666", marginBottom: "40px" }}>
            Connect to your database and start asking questions
          </p>
          
          <button
            onClick={fetchAgents}
            style={{
              width: "100%",
              padding: "16px",
              marginBottom: "16px",
              background: "#722ed1",
              color: "white",
              border: "none",
              borderRadius: "8px",
              fontSize: "16px",
              fontWeight: "600",
              cursor: "pointer",
              transition: "all 0.3s"
            }}
            onMouseOver={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
            onMouseOut={(e) => e.currentTarget.style.transform = "translateY(0)"}
          >
            📋 Select Existing Agent
          </button>
          
          <button
            onClick={() => setAppState("new-connection")}
            style={{
              width: "100%",
              padding: "16px",
              background: "#1890ff",
              color: "white",
              border: "none",
              borderRadius: "8px",
              fontSize: "16px",
              fontWeight: "600",
              cursor: "pointer",
              transition: "all 0.3s"
            }}
            onMouseOver={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
            onMouseOut={(e) => e.currentTarget.style.transform = "translateY(0)"}
          >
            🔗 Connect to New Database
          </button>
        </div>
      </div>
    );
  }

  // Render agent selection screen
  if (appState === "agent-selection") {
    return (
      <div style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        padding: "40px 20px"
      }}>
        <div style={{
          maxWidth: "800px",
          margin: "0 auto",
          background: "white",
          borderRadius: "16px",
          boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
          padding: "40px"
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "30px" }}>
            <h2 style={{ margin: 0, color: "#333" }}>Select an Agent</h2>
            <button
              onClick={() => setAppState("initial")}
              style={{
                padding: "8px 16px",
                background: "#f0f0f0",
                border: "none",
                borderRadius: "6px",
                cursor: "pointer"
              }}
            >
              ← Back
            </button>
          </div>

          {agents.length === 0 ? (
            <div style={{ textAlign: "center", padding: "40px", color: "#999" }}>
              <p>No agents found. Create a new connection first.</p>
            </div>
          ) : (
            <div style={{ display: "grid", gap: "16px" }}>
              {agents.map((agent) => (
                <div
                  key={agent.id}
                  style={{
                    border: "2px solid #e8e8e8",
                    borderRadius: "12px",
                    padding: "20px",
                    cursor: "pointer",
                    transition: "all 0.3s",
                    background: "white"
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.borderColor = "#1890ff";
                    e.currentTarget.style.boxShadow = "0 4px 12px rgba(24,144,255,0.15)";
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.borderColor = "#e8e8e8";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                  onClick={() => handleSelectAgent(agent)}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start" }}>
                    <div style={{ flex: 1 }}>
                      <h3 style={{ margin: "0 0 8px 0", color: "#333" }}>{agent.name}</h3>
                      {agent.description && (
                        <p style={{ margin: "0 0 12px 0", color: "#666", fontSize: "14px" }}>
                          {agent.description}
                        </p>
                      )}
                      <div style={{ fontSize: "12px", color: agent.trained_at ? "#52c41a" : "#ff4d4f" }}>
                        {agent.trained_at ? "✅ Trained" : "⚠️ Not trained"}
                      </div>
                    </div>
                    <button
                      style={{
                        padding: "8px 20px",
                        background: "#1890ff",
                        color: "white",
                        border: "none",
                        borderRadius: "6px",
                        fontWeight: "600",
                        cursor: "pointer"
                      }}
                    >
                      Connect
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Render new connection screen
  if (appState === "new-connection") {
    return (
      <div style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        padding: "40px 20px"
      }}>
        <div style={{
          maxWidth: "600px",
          margin: "0 auto",
          background: "white",
          borderRadius: "16px",
          boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
          padding: "40px"
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "30px" }}>
            <h2 style={{ margin: 0, color: "#333" }}>Connect to Database</h2>
            <button
              onClick={() => setAppState("initial")}
              style={{
                padding: "8px 16px",
                background: "#f0f0f0",
                border: "none",
                borderRadius: "6px",
                cursor: "pointer"
              }}
            >
              ← Back
            </button>
          </div>

          <div style={{ marginBottom: "20px" }}>
            <label style={{ display: "block", marginBottom: "8px", fontWeight: "600", color: "#333" }}>
              Database Type
            </label>
            <select
              value={dbType}
              onChange={(e) => setDbType(e.target.value)}
              disabled={connected}
              style={{
                width: "100%",
                padding: "12px",
                border: "2px solid #e8e8e8",
                borderRadius: "8px",
                fontSize: "14px"
              }}
            >
              <option value="postgresql">PostgreSQL</option>
              <option value="mysql">MySQL</option>
              <option value="sqlserver">SQL Server</option>
            </select>
          </div>

          <div style={{ marginBottom: "20px" }}>
            <label style={{ display: "block", marginBottom: "8px", fontWeight: "600", color: "#333" }}>Host</label>
            <input
              value={host}
              onChange={(e) => setHost(e.target.value)}
              placeholder="localhost"
              disabled={connected}
              style={{
                width: "100%",
                padding: "12px",
                border: "2px solid #e8e8e8",
                borderRadius: "8px",
                fontSize: "14px"
              }}
            />
          </div>

          <div style={{ marginBottom: "20px" }}>
            <label style={{ display: "block", marginBottom: "8px", fontWeight: "600", color: "#333" }}>Database Name</label>
            <input
              value={dbName}
              onChange={(e) => setDbName(e.target.value)}
              placeholder="my_database"
              disabled={connected}
              style={{
                width: "100%",
                padding: "12px",
                border: "2px solid #e8e8e8",
                borderRadius: "8px",
                fontSize: "14px"
              }}
            />
          </div>

          <div style={{ marginBottom: "20px" }}>
            <label style={{ display: "block", marginBottom: "8px", fontWeight: "600", color: "#333" }}>Username</label>
            <input
              value={user}
              onChange={(e) => setUser(e.target.value)}
              placeholder="username"
              disabled={connected}
              style={{
                width: "100%",
                padding: "12px",
                border: "2px solid #e8e8e8",
                borderRadius: "8px",
                fontSize: "14px"
              }}
            />
          </div>

          <div style={{ marginBottom: "20px" }}>
            <label style={{ display: "block", marginBottom: "8px", fontWeight: "600", color: "#333" }}>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="********"
              disabled={connected}
              style={{
                width: "100%",
                padding: "12px",
                border: "2px solid #e8e8e8",
                borderRadius: "8px",
                fontSize: "14px"
              }}
            />
          </div>

          <div style={{ marginBottom: "20px" }}>
            <label style={{ display: "block", marginBottom: "8px", fontWeight: "600", color: "#333" }}>Port</label>
            <input
              value={port}
              onChange={(e) => setPort(e.target.value)}
              placeholder="5432"
              disabled={connected}
              style={{
                width: "100%",
                padding: "12px",
                border: "2px solid #e8e8e8",
                borderRadius: "8px",
                fontSize: "14px"
              }}
            />
          </div>

          <button
            onClick={handleTestConnection}
            disabled={connected}
            style={{
              width: "100%",
              padding: "14px",
              background: connected ? "#52c41a" : "#1890ff",
              color: "white",
              border: "none",
              borderRadius: "8px",
              fontSize: "16px",
              fontWeight: "600",
              cursor: connected ? "default" : "pointer",
              marginBottom: "16px"
            }}
          >
            {connected ? "✅ Connected" : "Test Connection"}
          </button>

          {connectionMsg && (
            <div style={{
              padding: "12px",
              borderRadius: "8px",
              background: connected ? "#f6ffed" : "#fff2f0",
              border: `2px solid ${connected ? "#b7eb8f" : "#ffccc7"}`,
              color: connected ? "#389e0d" : "#cf1322",
              marginBottom: "20px",
              fontWeight: "600"
            }}>
              {connectionMsg}
            </div>
          )}

          {connected && showSaveAgent && (
            <div style={{
              padding: "24px",
              background: "#f0f5ff",
              border: "2px solid #91d5ff",
              borderRadius: "12px",
              marginTop: "20px"
            }}>
              <h3 style={{ margin: "0 0 20px 0", color: "#1890ff" }}>💾 Save as Agent</h3>
              
              <div style={{ marginBottom: "16px" }}>
                <label style={{ display: "block", marginBottom: "8px", fontWeight: "600", color: "#333" }}>
                  Agent Name *
                </label>
                <input
                  value={agentName}
                  onChange={(e) => setAgentName(e.target.value)}
                  placeholder="My Database Agent"
                  style={{
                    width: "100%",
                    padding: "12px",
                    border: "2px solid #d9d9d9",
                    borderRadius: "8px",
                    fontSize: "14px"
                  }}
                />
              </div>

              <div style={{ marginBottom: "16px" }}>
                <label style={{ display: "block", marginBottom: "8px", fontWeight: "600", color: "#333" }}>
                  Description (optional)
                </label>
                <textarea
                  value={agentDesc}
                  onChange={(e) => setAgentDesc(e.target.value)}
                  placeholder="Describe what this agent is for..."
                  rows={3}
                  style={{
                    width: "100%",
                    padding: "12px",
                    border: "2px solid #d9d9d9",
                    borderRadius: "8px",
                    fontSize: "14px",
                    resize: "vertical"
                  }}
                />
              </div>

              <button
                onClick={handleSaveAgent}
                disabled={!agentName.trim()}
                style={{
                  width: "100%",
                  padding: "12px",
                  background: agentName.trim() ? "#52c41a" : "#d9d9d9",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  fontSize: "16px",
                  fontWeight: "600",
                  cursor: agentName.trim() ? "pointer" : "not-allowed"
                }}
              >
                Save Agent
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Render connected/chat screen
  return (
    <div style={{ display: "flex", height: "100vh", background: "#f5f5f5" }}>
      {/* Sidebar */}
      <div style={{
        width: "320px",
        background: "white",
        borderRight: "1px solid #e8e8e8",
        display: "flex",
        flexDirection: "column",
        overflowY: "auto"
      }}>
        <div style={{ padding: "20px", borderBottom: "1px solid #e8e8e8" }}>
          <h3 style={{ margin: "0 0 8px 0", color: "#333" }}>
            {selectedAgent?.name}
          </h3>
          <p style={{ margin: "0 0 12px 0", fontSize: "12px", color: "#666" }}>
            {selectedAgent?.description || "No description"}
          </p>
          <button
            onClick={handleDisconnect}
            style={{
              width: "100%",
              padding: "8px",
              background: "#ff4d4f",
              color: "white",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
              fontSize: "14px",
              fontWeight: "600"
            }}
          >
            🔌 Disconnect
          </button>
        </div>

        {/* Training Section */}
        <div style={{ padding: "20px", borderBottom: "1px solid #e8e8e8" }}>
          <h4 style={{ margin: "0 0 12px 0", color: "#333" }}>🚀 Training</h4>
          
          {selectedAgent?.trained_at ? (
            <div style={{
              padding: "12px",
              background: "#f6ffed",
              border: "1px solid #b7eb8f",
              borderRadius: "6px",
              fontSize: "13px",
              color: "#52c41a",
              marginBottom: "12px"
            }}>
              ✅ Agent is trained and ready
            </div>
          ) : (
            <div style={{
              padding: "12px",
              background: "#fff7e6",
              border: "1px solid #ffd591",
              borderRadius: "6px",
              fontSize: "13px",
              color: "#fa8c16",
              marginBottom: "12px"
            }}>
              ⚠️ Agent needs training
            </div>
          )}

          <button
            onClick={handleTrainAgent}
            disabled={trainStatus === "running" || trainStatus === "queued"}
            style={{
              width: "100%",
              padding: "10px",
              background: (trainStatus === "running" || trainStatus === "queued") ? "#d9d9d9" : "#fa8c16",
              color: "white",
              border: "none",
              borderRadius: "6px",
              cursor: (trainStatus === "running" || trainStatus === "queued") ? "not-allowed" : "pointer",
              fontWeight: "600"
            }}
          >
            {trainStatus === "running" ? "Training..." : trainStatus === "queued" ? "Queued..." : "Train Agent"}
          </button>

          {trainRunId && (
            <div style={{ marginTop: "12px" }}>
              <div style={{ fontSize: "11px", marginBottom: "6px", color: "#666", textTransform: "uppercase" }}>
                {trainStatus} - {trainMsg}
              </div>
              <div style={{
                height: "8px",
                background: "#f0f0f0",
                borderRadius: "4px",
                overflow: "hidden"
              }}>
                <div style={{
                  width: `${trainProgress}%`,
                  height: "100%",
                  background: trainStatus === "failed" ? "#ff4d4f" : "#52c41a",
                  transition: "width 0.3s ease"
                }} />
              </div>
              <div style={{ fontSize: "10px", marginTop: "4px", color: "#999", textAlign: "right" }}>
                {trainProgress}%
              </div>
            </div>
          )}
        </div>

        {/* Conversation History */}
        {conversations.length > 0 && (
          <div style={{ flex: 1, padding: "20px", overflowY: "auto" }}>
            <h4 style={{ margin: "0 0 12px 0", color: "#333" }}>History ({conversations.length})</h4>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {conversations.slice().reverse().map((conv, index) => {
                const actualIndex = conversations.length - 1 - index;
                const userMsg = conv.messages.find(m => m.role === "user");
                const isSelected = currentConversationIndex === actualIndex;
                return (
                  <div
                    key={conv.session_id}
                    onClick={() => {
                      setCurrentConversationIndex(actualIndex);
                      loadConversationData(actualIndex);
                    }}
                    style={{
                      padding: "10px",
                      background: isSelected ? "#e6f7ff" : "#fafafa",
                      border: `1px solid ${isSelected ? "#91d5ff" : "#e8e8e8"}`,
                      borderRadius: "6px",
                      cursor: "pointer",
                      fontSize: "13px",
                      transition: "all 0.2s"
                    }}
                    onMouseOver={(e) => !isSelected && (e.currentTarget.style.background = "#f0f0f0")}
                    onMouseOut={(e) => !isSelected && (e.currentTarget.style.background = "#fafafa")}
                  >
                    <div style={{ fontWeight: "600", marginBottom: "4px", color: "#1890ff" }}>
                      #{index + 1}
                    </div>
                    <div style={{ 
                      overflow: "hidden", 
                      textOverflow: "ellipsis", 
                      whiteSpace: "nowrap",
                      color: "#666"
                    }}>
                      {userMsg?.content.slice(0, 50)}...
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Main Chat Area */}
      <div style={{ 
        flex: 1, 
        display: "flex", 
        flexDirection: "column",
        background: "#fafafa"
      }}>
        {/* Header */}
        <div style={{
          padding: "20px 30px",
          background: "white",
          borderBottom: "1px solid #e8e8e8"
        }}>
          <h2 style={{ margin: "0 0 4px 0", color: "#333" }}>Chat with AI Agent</h2>
          <p style={{ margin: 0, fontSize: "14px", color: "#666" }}>
            Ask questions about your data in natural language
          </p>
        </div>

        {/* Chat Content */}
        <div style={{ 
          flex: 1, 
          padding: "30px", 
          overflowY: "auto",
          display: "flex",
          flexDirection: "column"
        }}>
          {/* Question Input Area */}
          <div style={{
            background: "white",
            padding: "20px",
            borderRadius: "12px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
            marginBottom: "20px"
          }}>
            <textarea
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about your data..."
              disabled={loading || !isQuestionEditable}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleAsk();
                }
              }}
              style={{
                width: "100%",
                padding: "12px",
                border: "2px solid #e8e8e8",
                borderRadius: "8px",
                fontSize: "14px",
                resize: "vertical",
                fontFamily: "inherit",
                marginBottom: "12px"
              }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
              <div style={{ fontSize: "12px", color: "#999" }}>
                Press Enter to send, Shift+Enter for new line
              </div>
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  onClick={() => setIsQuestionEditable(!isQuestionEditable)}
                  style={{
                    padding: "8px 12px",
                    background: isQuestionEditable ? "#f0f0f0" : "#ffc53d",
                    color: "#333",
                    border: "none",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontWeight: "600",
                    fontSize: "12px"
                  }}
                >
                  {isQuestionEditable ? "Lock" : "Edit"}
                </button>
                <button
                  onClick={handleReset}
                  style={{
                    padding: "8px 12px",
                    background: "#ff4d4f",
                    color: "white",
                    border: "none",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontWeight: "600",
                    fontSize: "12px"
                  }}
                >
                  New Query
                </button>
                <button
                  onClick={handleAsk}
                  disabled={loading || !question.trim()}
                  style={{
                    padding: "10px 24px",
                    background: (loading || !question.trim()) ? "#d9d9d9" : "#1890ff",
                    color: "white",
                    border: "none",
                    borderRadius: "6px",
                    cursor: (loading || !question.trim()) ? "not-allowed" : "pointer",
                    fontWeight: "600",
                    fontSize: "14px"
                  }}
                >
                  {loading ? "Thinking..." : "Ask Question"}
                </button>
              </div>
            </div>
          </div>

          {/* Navigation Controls */}
          {conversations.length > 0 && (
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "12px",
              marginBottom: "20px",
              padding: "12px",
              background: "white",
              borderRadius: "8px",
              boxShadow: "0 2px 8px rgba(0,0,0,0.08)"
            }}>
              <button
                onClick={navigateToPrevious}
                disabled={currentConversationIndex >= conversations.length - 1}
                style={{
                  padding: "8px 16px",
                  background: currentConversationIndex >= conversations.length - 1 ? "#f5f5f5" : "#1890ff",
                  color: currentConversationIndex >= conversations.length - 1 ? "#ccc" : "white",
                  border: "none",
                  borderRadius: "6px",
                  cursor: currentConversationIndex >= conversations.length - 1 ? "not-allowed" : "pointer",
                  fontWeight: "600",
                  fontSize: "14px"
                }}
              >
                Previous
              </button>

              <div style={{
                padding: "8px 16px",
                background: "#f0f5ff",
                border: "2px solid #91d5ff",
                borderRadius: "6px",
                fontSize: "14px",
                fontWeight: "600",
                color: "#1890ff",
                minWidth: "100px",
                textAlign: "center"
              }}>
                {currentConversationIndex === -1 
                  ? `${conversations.length + 1} of ${conversations.length + 1}` 
                  : `${conversations.length - currentConversationIndex} of ${conversations.length + 1}`}
              </div>

              <button
                onClick={navigateToNext}
                disabled={currentConversationIndex === -1}
                style={{
                  padding: "8px 16px",
                  background: currentConversationIndex === -1 ? "#f5f5f5" : "#1890ff",
                  color: currentConversationIndex === -1 ? "#ccc" : "white",
                  border: "none",
                  borderRadius: "6px",
                  cursor: currentConversationIndex === -1 ? "not-allowed" : "pointer",
                  fontWeight: "600",
                  fontSize: "14px"
                }}
              >
                Next
              </button>
            </div>
          )}

          {/* Response Display */}
          {response && (
            <div style={{
              background: "white",
              borderRadius: "12px",
              boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
              overflow: "visible"
            }}>
              {/* SQL Query Section */}
              <div style={{ padding: "20px", borderBottom: "1px solid #f0f0f0" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <h3 style={{ margin: "0 0 12px 0", color: "#1890ff", fontSize: "16px" }}>
                    Generated SQL Query
                  </h3>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <button
                      onClick={() => setIsEditingSQL(!isEditingSQL)}
                      style={{
                        padding: "6px 12px",
                        background: isEditingSQL ? "#f0f0f0" : "#ffc53d",
                        color: "#333",
                        border: "none",
                        borderRadius: "6px",
                        cursor: "pointer",
                        fontWeight: "600",
                        fontSize: "12px"
                      }}
                    >
                      {isEditingSQL ? "Cancel" : "Edit"}
                    </button>
                    {isEditingSQL && (
                      <button
                        onClick={applyEditedSQL}
                        disabled={loading || !editedSQL.trim()}
                        style={{
                          padding: "6px 12px",
                          background: (loading || !editedSQL.trim()) ? "#d9d9d9" : "#52c41a",
                          color: "white",
                          border: "none",
                          borderRadius: "6px",
                          cursor: (loading || !editedSQL.trim()) ? "not-allowed" : "pointer",
                          fontWeight: "600",
                          fontSize: "12px"
                        }}
                      >
                        {loading ? "Running..." : "Apply"}
                      </button>
                    )}
                  </div>
                </div>
                {!isEditingSQL ? (
                  <pre style={{
                    background: "#f6f8fa",
                    padding: "16px",
                    borderRadius: "8px",
                    border: "1px solid #e1e4e8",
                    fontSize: "13px",
                    overflow: "auto",
                    whiteSpace: "pre-wrap",
                    margin: 0,
                    fontFamily: "'Monaco', 'Menlo', monospace"
                  }}>
                    {response.sql || "No SQL generated"}
                  </pre>
                ) : (
                  <textarea
                    rows={6}
                    value={editedSQL}
                    onChange={(e) => setEditedSQL(e.target.value)}
                    style={{
                      width: "100%",
                      padding: "12px",
                      border: "2px solid #d9d9d9",
                      borderRadius: "8px",
                      fontSize: "13px",
                      fontFamily: "'Monaco', 'Menlo', monospace",
                      marginTop: "12px"
                    }}
                  />
                )}
              </div>

              {/* Results Section */}
              {response.data && response.data.columns && response.data.columns.length > 0 && (
                <div style={{ padding: "20px", borderBottom: "1px solid #f0f0f0" }}>
                  <h3 style={{ margin: "0 0 12px 0", color: "#52c41a", fontSize: "16px" }}>
                    Query Results
                  </h3>
                  <div style={{
                    overflowX: "auto",
                    overflowY: "scroll",
                    WebkitOverflowScrolling: "touch",
                    border: "1px solid #e8e8e8",
                    borderRadius: "8px",
                    height: "20vh",
                    minHeight: "120px",
                    maxWidth: "100%",
                    position: "relative"
                  }}>
                    <table style={{
                      borderCollapse: "collapse",
                      width: "max-content",
                      minWidth: "100%",
                      fontSize: "13px",
                      tableLayout: "auto"
                    }}>
                      <thead>
                        <tr style={{ background: "#fafafa" }}>
                          {response.data.columns.map((col) => (
                            <th key={col} style={{
                              border: "1px solid #e8e8e8",
                              padding: "10px 12px",
                              textAlign: "left",
                              fontWeight: "600",
                              position: "sticky",
                              top: 0,
                              background: "#fafafa",
                              zIndex: 1,
                              whiteSpace: "nowrap"
                            }}>
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {response.data.rows.map((row, i) => (
                          <tr key={i} style={{ background: i % 2 === 0 ? "white" : "#fafafa" }}>
                            {response.data.columns.map((col) => (
                              <td
                                key={col}
                                title={String(row[col] ?? "")}
                                style={{
                                  border: "1px solid #e8e8e8",
                                  padding: "8px 12px",
                                  whiteSpace: "nowrap"
                                }}>
                                {String(row[col] ?? "")}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ marginTop: "8px", fontSize: "12px", color: "#999" }}>
                    Showing {response.data.rows.length} row(s)
                  </div>
                </div>
              )}

              {/* Feedback Section - Only for current conversation */}
              {currentConversationIndex === -1 && (
                <div style={{ padding: "20px", borderBottom: "1px solid #f0f0f0", background: "#fafafa" }}>
                  <h4 style={{ margin: "0 0 12px 0", color: "#666", fontSize: "14px" }}>
                    Was this response helpful?
                  </h4>
                  <div style={{ display: "flex", gap: "12px" }}>
                    <button
                      onClick={() => sendFeedback(true)}
                      style={{
                        padding: "10px 20px",
                        background: "#52c41a",
                        color: "white",
                        border: "none",
                        borderRadius: "6px",
                        cursor: "pointer",
                        fontWeight: "600",
                        fontSize: "14px"
                      }}
                    >
                      Correct
                    </button>
                    <button
                      onClick={() => sendFeedback(false)}
                      style={{
                        padding: "10px 20px",
                        background: "#ff4d4f",
                        color: "white",
                        border: "none",
                        borderRadius: "6px",
                        cursor: "pointer",
                        fontWeight: "600",
                        fontSize: "14px"
                      }}
                    >
                      Incorrect
                    </button>
                  </div>

                  {feedback && (
                    <div style={{
                      marginTop: "12px",
                      padding: "10px 12px",
                      background: feedback.includes("correct") ? "#f6ffed" : "#fff2f0",
                      border: `1px solid ${feedback.includes("correct") ? "#b7eb8f" : "#ffccc7"}`,
                      borderRadius: "6px",
                      fontSize: "13px",
                      color: feedback.includes("correct") ? "#389e0d" : "#cf1322"
                    }}>
                      {feedback}
                    </div>
                  )}
                </div>
              )}

              {/* Answer Section */}
              <div style={{ padding: "20px" }}>
                <h3 style={{ margin: "0 0 12px 0", color: "#722ed1", fontSize: "16px" }}>
                  Answer
                </h3>
                <div style={{
                  background: "#f9f0ff",
                  padding: "16px",
                  borderRadius: "8px",
                  border: "1px solid #d3adf7",
                  fontSize: "14px",
                  lineHeight: "1.6",
                  color: "#333"
                }}>
                  {response.answer}
                </div>
              </div>
            </div>
          )}

          {/* Empty State */}
          {!response && !loading && (
            <div style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              textAlign: "center",
              color: "#999",
              padding: "40px"
            }}>
              <div>
                <div style={{ fontSize: "48px", marginBottom: "16px" }}>💬</div>
                <h3 style={{ margin: "0 0 8px 0", color: "#666" }}>Start a Conversation</h3>
                <p style={{ margin: 0 }}>Ask a question about your data to get started</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Chat;