const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { PythonShell } = require('python-shell');
const path = require('path');

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Serve static files (for frontend)
app.use(express.static(path.join(__dirname, '../frontend')));

// Prediction endpoint
app.post('/predict', (req, res) => {
  const inputData = req.body;

  const options = {
    scriptPath: __dirname, // Points to the backend folder
    pythonPath: 'python',  // Use 'python3' if on Linux/Mac
    pythonOptions: ['-u'], // Unbuffered output
    args: [JSON.stringify(inputData)]
  };

  PythonShell.run('predict.py', options, (err, results) => {
    if (err) {
      console.error('PythonShell Error:', err);
      return res.status(500).json({ error: 'Prediction failed', details: err.message });
    }
    
    try {
      const prediction = results[0];
      res.json({ prediction: parseInt(prediction) }); // Ensure number output
    } catch (e) {
      console.error('Output Parse Error:', e);
      res.status(500).json({ error: 'Invalid prediction output' });
    }
  });
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});