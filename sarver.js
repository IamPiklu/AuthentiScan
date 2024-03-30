const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const app = express();

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

app.use(express.static('public'));  // Serve your HTML file

app.post('/upload', upload.single('image'), (req, res) => {
  const python = spawn('python', ['app.py', req.file.path]);

  python.stdout.on('data', (data) => {
    const predictions = JSON.parse(data);
    const { real, fake } = predictions;
    res.json({ real: real * 100, fake: fake * 100 });
    console.log(`real: ${(real*100).toFixed(2)}%, fake: ${(fake*100).toFixed(2)}%`);
  });

  python.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });
});

app.listen(3000, () => console.log('Server started on port 3000'));
