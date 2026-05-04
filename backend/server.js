const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();

// Increase payload limit to allow receiving all strategies and their equity curves
app.use(express.json({ limit: '50mb' }));
app.use(cors());

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// In-memory data store
let latestData = {
    status: {
        running: false,
        strategies_tested: 0,
        best_score: 0.0,
        best_win_rate: 0.0,
        current_status: "Waiting for data from Python engine...",
        errors: 0,
        leaderboard_size: 0
    },
    leaderboard: [],
    equities: {}
};

// Receive data from Python engine
app.post('/api/data', (req, res) => {
    try {
        const { status, leaderboard, equities } = req.body;
        
        if (status) latestData.status = status;
        if (leaderboard) latestData.leaderboard = leaderboard;
        if (equities) {
            // Merge equities to avoid losing history if a strategy drops out of the current batch occasionally
            latestData.equities = { ...latestData.equities, ...equities };
        }
        
        console.log(`[${new Date().toISOString()}] Received data payload from Python engine. LB Size: ${leaderboard?.length || 0}`);
        res.json({ success: true });
    } catch (err) {
        console.error("Error parsing incoming data:", err);
        res.status(400).json({ success: false, error: "Invalid data format" });
    }
});

// Provide status to frontend
app.get('/api/status', (req, res) => {
    res.json(latestData.status);
});

// Provide leaderboard to frontend
app.get('/api/leaderboard', (req, res) => {
    res.json(latestData.leaderboard);
});

// Provide equity curve for a specific strategy
app.get('/api/equity/:id', (req, res) => {
    const stratId = req.params.id;
    if (latestData.equities && latestData.equities[stratId]) {
        res.json(latestData.equities[stratId]);
    } else {
        res.json([]);
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Dashboard server running on port ${PORT}`);
    console.log(`Waiting for Python POST requests at http://localhost:${PORT}/api/data`);
});
