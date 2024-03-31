const express = require("express");
const multer = require("multer");
const { spawn } = require("child_process");
const path = require("path");
const app = express();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // Use the mime type of the file to determine the folder
    let destFolder = "uploads/";
    if (file.mimetype.startsWith("image/")) {
      destFolder += "images/";
    } else if (file.mimetype.startsWith("video/")) {
      destFolder += "videos/";
    }
    cb(null, destFolder);
  },
  filename: function (req, file, cb) {
    let prefix = file.mimetype.startsWith("image/") ? "image" : "video";
    cb(null, prefix + "-" + Date.now() + path.extname(file.originalname));
  },
});

const fileFilter = (req, file, cb) => {
  // Accept images and videos only
  if (
    file.mimetype.startsWith("image/") ||
    file.mimetype.startsWith("video/")
  ) {
    cb(null, true);
  } else {
    cb(new Error("Only image and video files are allowed!"), false);
  }
};

const upload = multer({ storage: storage, fileFilter: fileFilter });

app.use(express.static("public")); // Serve your HTML file

app.post("/upload", upload.single("image"), (req, res) => {
  const python = spawn("python", ["app.py", req.file.path]);

  python.stdout.on("data", (data) => {
    const predictions = JSON.parse(data);
    const { real, fake } = predictions;
    res.json({ real: real * 100, fake: fake * 100 });
    console.log(
      `real: ${(real * 100).toFixed(2)}%, fake: ${(fake * 100).toFixed(2)}%`
    );
  });

  python.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on("close", (code) => {
    console.log(`child process exited with code ${code}`);
  });
});

app.listen(3000, () =>
  console.log("Server started on port http://localhost:3000")
);
