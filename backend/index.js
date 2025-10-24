import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import axios from "axios";
import pkg from "pg";
import { v4 as uuidv4 } from "uuid";
import dotenv from "dotenv";

const { Pool } = pkg;
const app = express();
app.use(cors());
app.use(bodyParser.json());

dotenv.config();

const pool = new Pool({
  connectionString: process.env.APP_DATABASE_URL,
  ssl: { rejectUnauthorized: false }
});


const PYTHON_AI_URL = process.env.PYTHON_AI_URL || "http://127.0.0.1:8000";
const DEFAULT_USER_ID = process.env.DEFAULT_USER_ID || '1';

// Health check for Python service with improved timeout and retry logic
// Health check for Python service with improved timeout and retry logic
let pythonServiceHealthy = false;
let healthCheckAttempts = 0;
const MAX_HEALTH_CHECK_ATTEMPTS = 3;

const checkPythonService = async () => {
  try {
    healthCheckAttempts++;
    
    // Progressive timeout - start with shorter timeout, increase if needed
    const timeout = Math.min(2000 + (healthCheckAttempts * 1000), 8000);
    
    console.log(`Health check attempt ${healthCheckAttempts}/${MAX_HEALTH_CHECK_ATTEMPTS} with ${timeout}ms timeout`);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const response = await axios.get(`${PYTHON_AI_URL}/ping`, { 
      timeout: timeout,
      signal: controller.signal,
      headers: {
        'Connection': 'close' // Prevent keep-alive issues
      }
    });
    
    clearTimeout(timeoutId);
    
    if (response.status === 200) {
      debugger;
      pythonServiceHealthy = true;
      healthCheckAttempts = 0; // Reset attempts on success
      console.log("✅ Python service health check successful");
    } else {
      throw new Error(`Unexpected status: ${response.status}`);
    }
    
  } catch (err) {
    pythonServiceHealthy = false;
    
    if (err.code === 'ECONNREFUSED') {
      console.error("❌ Python service is not running or not accessible");
    } else if (err.code === 'ECONNABORTED' || err.name === 'AbortError') {
      console.error(`⏰ Python service health check timed out after ${err.timeout || 'unknown'}ms`);
    } else if (err.code === 'ENOTFOUND') {
      console.error("❌ Python service host not found - check PYTHON_AI_URL");
    } else {
      console.error("❌ Python service health check failed:", err.message);
    }
    
    // Reset attempts if we've exceeded max attempts
    if (healthCheckAttempts >= MAX_HEALTH_CHECK_ATTEMPTS) {
      console.log(`Resetting health check attempts after ${MAX_HEALTH_CHECK_ATTEMPTS} failures`);
      healthCheckAttempts = 0;
    }
  }
};

// Initial health check with delay to allow Python service to start
setTimeout(() => {
  checkPythonService();
}, 2000);

// Regular health checks - less frequent to reduce load
setInterval(checkPythonService, 45000); // Every 45 seconds instead of 30

// Improved middleware to check Python service health
const requirePythonService = (req, res, next) => {
  if (!pythonServiceHealthy) {
    // Try one quick health check before failing
    checkPythonService();
    
    return res.status(503).json({ 
      error: "AI service temporarily unavailable",
      details: "Python FastAPI service is not responding. Please ensure the Python service is running on port 8000.",
      retry_after: 30
    });
  }
  next();
};

// Enhanced error handler for axios requests with better timeout handling
const handlePythonServiceRequest = async (req, res, url, body = null, method = 'GET') => {
  try {
    let response;
    
    // Different timeouts for different operations
    const timeouts = {
      'GET': 10000,      // 10 seconds for GET requests
      'POST': 45000,     // 45 seconds for POST requests (training, etc.)
      'test-connection': 15000,  // 15 seconds for connection tests
      'train': 120000    // 2 minutes for training operations
    };
    
    // Determine appropriate timeout based on URL and method
    let timeout = timeouts[method] || 30000;
    
    if (url.includes('/test-connection')) {
      timeout = timeouts['test-connection'];
    } else if (url.includes('/train/')) {
      timeout = timeouts['train'];
    }
    
    console.log(`Making ${method} request to ${url} with ${timeout}ms timeout`);
    
    const config = { 
      timeout: timeout,
      headers: {
        'Content-Type': 'application/json',
        'Connection': 'close'
      },
      // Add retry logic for failed requests
      validateStatus: function (status) {
        return status < 500; // Don't throw for 4xx errors, handle them gracefully
      }
    };
    
    const startTime = Date.now();
    
    if (method === 'POST') {
      response = await axios.post(url, body || req.body, config);
    } else {
      response = await axios.get(url, config);
    }
    
    const duration = Date.now() - startTime;
    const size = typeof response.data === 'string' ? response.data.length : JSON.stringify(response.data).length;
    console.log(`Request completed in ${duration}ms, status=${response.status}, bytes=${size}`);
    
    // Handle different response statuses
    if (response.status >= 400) {
      return res.status(response.status).json(response.data);
    }
    
    res.json(response.data);
    
  } catch (err) {
    const duration = Date.now() - (err.startTime || Date.now());
    console.error(`Error ${method} ${url} (${duration}ms):`, {
      code: err.code,
      message: err.message,
      status: err.response?.status,
      data: err.response?.data
    });
    
    // More specific error handling
    if (err.code === 'ECONNREFUSED') {
      pythonServiceHealthy = false;
      res.status(503).json({ 
        error: "AI service unavailable",
        details: "Cannot connect to Python service. Please ensure it's running."
      });
    } else if (err.code === 'ECONNABORTED' || err.message.includes('timeout')) {
      res.status(504).json({ 
        error: "Request timeout",
        details: `Request took longer than ${err.timeout || 'expected'} milliseconds to complete`
      });
    } else if (err.response) {
      res.status(err.response.status).json(err.response.data);
    } else if (err.code === 'ENOTFOUND') {
      res.status(503).json({ 
        error: "Service configuration error",
        details: "Cannot resolve Python service hostname"
      });
    } else {
      res.status(500).json({ 
        error: "Internal server error",
        details: err.message 
      });
    }
  }
};

// Test DB connection with enhanced SQL Server support
app.post("/api/test-connection", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/test-connection`, req.body, 'POST');
});

// Validate connection string (useful for SQL Server)
app.post("/api/validate-connection-string", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/validate-connection-string`, req.body, 'POST');
});

// Test SQL Server drivers
app.get("/api/test-sqlserver-drivers", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/test-sqlserver-drivers`);
});

// Ask AI agent
app.post("/api/ask", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/ask`, req.body, 'POST');
});

// Feedback loop
app.post("/api/feedback", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/feedback`, req.body, 'POST');
});

// Execute edited SQL (manual run)
app.post("/api/execute", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/execute`, req.body, 'POST');
});



// Save Agent - Improved version with better error handling
app.post(
  "/api/agents",
  requirePythonService,
  async (req, res) => {
    await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/agents`, req.body, "POST" );
  }
);


/*
app.post("/api/agents", async (req, res) => {
  try {
    const {
      user_id,
      name,
      description,
      db_url,
      host,
      dbName,
      user,
      password,
      port,
      dbType
    } = req.body;

    // Enhanced validation
    if (!user_id || !name) {
      return res.status(400).json({ 
        success: false, 
        error: "user_id and name are required" 
      });
    }

    // Validate name length and characters (prevent SQL issues)
    if (name.length > 255) {
      return res.status(400).json({
        success: false,
        error: "Agent name must be less than 255 characters"
      });
    }

    // Build db_url with proper encoding for special characters
    let finalDbUrl = db_url;
    
    if (dbType === "sqlserver" || dbType === "mysql") {
      // More robust encoding for passwords with special characters
      const encodedUser = encodeURIComponent(user || "");
      const encodedPassword = encodeURIComponent(password || "");
      
      console.log("Original password:", password);
      console.log("Encoded password:", encodedPassword);
      
      if (dbType === "sqlserver") {
        // SQL Server connection string - handle special characters properly
        finalDbUrl = `mssql+pyodbc://${encodedUser}:${encodedPassword}@${host || ""}:${port || 1433}/${dbName || ""}?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes&Encrypt=no`;
      } else if (dbType === "mysql") {
        // MySQL connection string
        finalDbUrl = `mysql://${encodedUser}:${encodedPassword}@${host || ""}:${port || 3306}/${dbName || ""}`;
      }
    } else {
      // PostgreSQL - also encode properly
      const encodedUser = encodeURIComponent(user || "");
      const encodedPassword = encodeURIComponent(password || "");
      finalDbUrl = `postgresql://${encodedUser}:${encodedPassword}@${host || ""}:${port || 5432}/${dbName || ""}`;
    }

    console.log("Final DB URL (without password):", finalDbUrl.replace(/:([^:@]+)@/, ':****@'));

    const id = uuidv4();

    // Enhanced database query with better error handling
    try {
      await pool.query(
        `INSERT INTO ai_agents (id, user_id, name, description, db_url)
         VALUES ($1, $2, $3, $4, $5)`,
        [id, user_id, name, description || null, finalDbUrl]
      );

      console.log(`Successfully saved agent: ${name} with ID: ${id}`);
      res.json({ success: true, id });

    } catch (dbError) {
      console.error("Database insertion error:", {
        code: dbError.code,
        message: dbError.message,
        detail: dbError.detail,
        constraint: dbError.constraint,
        table: dbError.table,
        column: dbError.column
      });

      // Handle specific database errors
      if (dbError.code === '23505') {
        return res.status(409).json({ 
          success: false, 
          error: "Agent with this name already exists for this user" 
        });
      } else if (dbError.code === '23502') {
        return res.status(400).json({
          success: false,
          error: `Missing required field: ${dbError.column}`
        });
      } else if (dbError.code === '22001') {
        return res.status(400).json({
          success: false,
          error: "One of the fields is too long for the database"
        });
      } else {
        throw dbError; // Re-throw to be caught by outer catch
      }
    }

  } catch (err) {
    console.error("Error /api/agents:", {
      message: err.message,
      stack: err.stack,
      code: err.code
    });
    
    res.status(500).json({ 
      success: false, 
      error: "Failed to save agent: " + err.message 
    });
  }
});
*/

// ------------------ AI Agent Management ------------------
/*
// Save Agent
app.post("/api/agents", async (req, res) => {
  try {
    const {
      user_id,
      name,
      description,
      db_url,
      host,
      dbName,
      user,
      password,
      port,
      dbType
    } = req.body;

    // Validate required fields
    if (!user_id || !name) {
      return res.status(400).json({ 
        success: false, 
        error: "user_id and name are required" 
      });
    }

    // Build db_url with enhanced SQL Server support
    let finalDbUrl = db_url;
    if (dbType === "sqlserver" || dbType === "mysql" ) {
      const encodedUser = encodeURIComponent(user || "");
      const encodedPassword = encodeURIComponent(password || "");
      if (dbType === "sqlserver") {
        // SQL Server (pyodbc format for Vanna)
        finalDbUrl = `mssql+pyodbc://${encodedUser}:${encodedPassword}@${host || ""}:${port || 1433}/${dbName || ""}?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes`;
      } else if (dbType === "mysql") {
        // MySQL uses a different scheme
        finalDbUrl = `mysql://${encodedUser}:${encodedPassword}@${host || ""}:${port || 3306}/${dbName || ""}`;
      } else {
        // Default: PostgreSQL
        finalDbUrl = `${dbType || "postgresql"}://${encodedUser}:${encodedPassword}@${host || ""}:${port || 5432}/${dbName || ""}`;
      }
    }

    const id = uuidv4();

    await pool.query(
      `INSERT INTO ai_agents (id, user_id, name, description, db_url)
       VALUES ($1, $2, $3, $4, $5)`,
      [id, user_id, name, description, finalDbUrl]
    );

    res.json({ success: true, id });
  } catch (err) {
    console.error("Error /api/agents:", err.message);
    
    if (err.code === '23505') {
      res.status(409).json({ success: false, error: "Agent with this name already exists" });
    } else {
      res.status(500).json({ success: false, error: err.message });
    }
  }
});

*/
// List Agents (supports optional ?user_id=...; falls back to DEFAULT_USER_ID)
app.get("/api/agents", requirePythonService, async (req, res) => {
  const raw = (req.query.user_id ?? DEFAULT_USER_ID) + '';
  const uid = /^[0-9]+$/.test(raw) ? raw : DEFAULT_USER_ID;
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/agents/${uid}`);
});

// Get Agents for a User
app.get("/api/agents/:user_id", requirePythonService, async (req, res) => {
  // Coerce non-numeric user IDs to 1 to match AI service app DB expectations
  const raw = req.params.user_id || '';
  const uid = /^[0-9]+$/.test(String(raw)) ? String(raw) : '1';
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/agents/${uid}`);
});

// Delete Agent
app.delete("/api/agents/:agent_id", async (req, res) => {
  try {
    const { agent_id } = req.params;
    const { user_id } = req.query;

    if (!agent_id || !user_id) {
      return res.status(400).json({ 
        success: false, 
        error: "agent_id and user_id are required" 
      });
    }

    const result = await pool.query(
      "DELETE FROM ai_agents WHERE id=$1 AND user_id=$2",
      [agent_id, user_id]
    );

    if (result.rowCount === 0) {
      return res.status(404).json({ 
        success: false, 
        error: "Agent not found or access denied" 
      });
    }

    res.json({ success: true });
  } catch (err) {
    console.error("Error deleting agent:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
});

// TRAIN endpoints
app.post("/api/train/start", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/train/start`, req.body, 'POST');
}); 

app.get("/api/train/status/:run_id", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/train/status/${req.params.run_id}`);
});

// Get conversation history
app.get("/api/history/:agent_id", requirePythonService, async (req, res) => {
  const { agent_id } = req.params;
  const { limit = 50 } = req.query;
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/history/${agent_id}?limit=${limit}`);
});

// Favourites: list by agent
app.get("/api/favorites/:agent_id", requirePythonService, async (req, res) => {
  const { agent_id } = req.params;
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/favorites/${agent_id}`);
});

// Favourites: detail by favorite_id
app.get("/api/favorites/detail/:favorite_id", requirePythonService, async (req, res) => {
  const { favorite_id } = req.params;
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/favorites/detail/${favorite_id}`);
});

// System prompts endpoints
app.post("/api/prompts", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/prompts`, req.body, 'POST');
});

// QnA endpoints  
app.post("/api/qna", requirePythonService, async (req, res) => {
  await handlePythonServiceRequest(req, res, `${PYTHON_AI_URL}/qna`, req.body, 'POST');
});

// Enhanced health check endpoint with more detailed service status
app.get("/api/health", async (req, res) => {
  try {
    // Check database connection
    const dbStart = Date.now();
    await pool.query('SELECT 1');
    const dbDuration = Date.now() - dbStart;
    
    // Check Python service with a quick health check
    let pythonHealth = {
      status: "disconnected",
      duration: null,
      error: null
    };
    
    try {
      const pythonStart = Date.now();
      const pythonResponse = await axios.get(`${PYTHON_AI_URL}/ping`, { 
        timeout: 5000,
        headers: { 'Connection': 'close' }
      });
      pythonHealth = {
        status: pythonResponse.data.status || "connected",
        duration: Date.now() - pythonStart,
        error: null
      };
    } catch (err) {
      pythonHealth = {
        status: "disconnected",
        duration: null,
        error: err.code || err.message
      };
    }
    
    const overallStatus = pythonHealth.status === "connected" ? "healthy" : "degraded";
    
    res.json({ 
      status: overallStatus,
      services: {
        database: {
          status: "connected",
          duration: dbDuration
        },
        python_ai: pythonHealth
      },
      environment: {
        node_version: process.version,
        python_url: PYTHON_AI_URL
      },
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    res.status(503).json({ 
      status: "unhealthy",
      error: err.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Add a specific endpoint to test Python service connectivity
app.get("/api/python-service/ping", async (req, res) => {
  try {
    const start = Date.now();
    const response = await axios.get(`${PYTHON_AI_URL}/health`, {
      timeout: 10000,
      headers: { 'Connection': 'close' }
    });
    const duration = Date.now() - start;
    
    res.json({
      success: true,
      duration: duration,
      python_response: response.data,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({
      success: false,
      error: error.message,
      code: error.code,
      timeout: error.timeout,
      timestamp: new Date().toISOString()
    });
  }
});

// Enhanced root route
app.get("/", (req, res) => {
  res.send(`
    <html>
      <head>
        <title>AI Service Backend</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            text-align: center; 
            margin: 0;
            padding: 50px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
          }
          .container {
            max-width: 600px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            padding: 40px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
          }
          h1 { 
            color: #fff;
            margin-bottom: 10px;
            font-size: 2.5em;
            font-weight: 300;
          }
          p { 
            font-size: 1.2em;
            margin: 20px 0;
            opacity: 0.9;
          }
          .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
          }
          .status-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
          }
          .status-indicator {
            font-size: 2em;
            margin-bottom: 10px;
          }
          .healthy { color: #4ade80; }
          .unhealthy { color: #ef4444; }
          .tag {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 8px 16px;
            border-radius: 25px;
            margin: 10px;
            font-weight: 500;
            font-size: 0.9em;
          }
          .features {
            text-align: left;
            margin: 30px 0;
          }
          .features ul {
            list-style: none;
            padding: 0;
          }
          .features li {
            padding: 8px 0;
            opacity: 0.9;
          }
          .features li:before {
            content: "✓ ";
            color: #4ade80;
            font-weight: bold;
            margin-right: 8px;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>AI SQL Chat Service</h1>
          <p>Enterprise-grade AI-powered SQL query assistant</p>
          
          <div class="status-grid">
            <div class="status-card">
              <div class="status-indicator ${pythonServiceHealthy ? 'healthy' : 'unhealthy'}">
                ${pythonServiceHealthy ? '🟢' : '🔴'}
              </div>
              <strong>Python AI Service</strong><br>
              ${pythonServiceHealthy ? 'Connected' : 'Disconnected'}
            </div>
            
            <div class="status-card">
              <div class="status-indicator healthy">🟢</div>
              <strong>Express Gateway</strong><br>
              Running on Port ${PORT}
            </div>
          </div>

          <div class="features">
            <h3>Features:</h3>
            <ul>
              <li>Multi-database support (PostgreSQL, MySQL, SQL Server)</li>
              <li>AI-powered SQL generation with OpenAI</li>
              <li>Enhanced SQL Server connectivity</li>
              <li>Conversation history and learning</li>
              <li>Real-time training progress</li>
              <li>Schema analysis and storage</li>
            </ul>
          </div>

          <div>
            <span class="tag">REST API Gateway</span>
            <span class="tag">AI/ML Ready</span>
            <span class="tag">Enterprise Scale</span>
          </div>
        </div>
      </body>
    </html>
  `);
});

// Global error handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ 
    success: false, 
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// Handle 404s
app.use((req, res) => {
  res.status(404).json({ 
    success: false, 
    error: `Route ${req.method} ${req.path} not found` 
  });
});

const PORT = process.env.PORT || 5000;

// Graceful shutdown
const server = app.listen(PORT, () => {
  console.log(`🚀 Express backend running on http://localhost:${PORT}`);
  console.log(`🔗 Python AI Service: ${pythonServiceHealthy ? '✅ Connected' : '❌ Disconnected'}`);
});

process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  server.close(() => {
    console.log('Express server closed');
    pool.end(() => {
      console.log('Database pool closed');
      process.exit(0);
    });
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully...');
  server.close(() => {
    console.log('Express server closed');
    pool.end(() => {
      console.log('Database pool closed');
      process.exit(0);
    });
  });
}); 