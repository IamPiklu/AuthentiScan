const { spawn } = require("child_process");
const python = spawn("python", ["app.py", "C:/Users/sayan/OneDrive/Pictures/Screenshots/fake_rashmika.png"]);

python.stdout.on("data", (data) => {
  const predictions = JSON.parse(data);
  const { real, fake } = predictions;
  // console.log(predictions);
  console.log(`real: ${(real * 100).toFixed(2)}%`);
  console.log(`fake: ${(fake * 100).toFixed(2)}%`);
});

python.stderr.on("data", (data) => {
  console.error(`stderr: ${data}`);
});

python.on("close", (code) => {
  // console.log(`child process exited with code ${code}`);
});
