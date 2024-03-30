document.querySelector('#imageInput').addEventListener('change', function() {
  const reader = new FileReader();

  reader.onload = function (e) {
    // Get the preview element
    const preview = document.querySelector('#preview');

    // Set the source of the image to the selected file
    preview.src = e.target.result;

    // Make the preview visible
    preview.style.display = 'block';

    // Hide the upload button
    document.querySelector('#uploadButton').style.display = 'none';
  };

  // Read the image file
  reader.readAsDataURL(this.files[0]);
});

document.querySelector('form').addEventListener('submit', (event) => {
  event.preventDefault();

  // Disable the form to prevent multiple submissions
  const form = document.querySelector('form');
  form.classList.add('opacity-50');
  form.classList.add('cursor-not-allowed');
  form.querySelectorAll('input').forEach(input => input.disabled = true);

  const fileInput = document.querySelector('#imageInput');
  const formData = new FormData();
  formData.append('image', fileInput.files[0]);

  fetch('/upload', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(predictions => {
    // Get the div element by id
    const divElement = document.querySelector('#container');

    // Create an image element for 'face_with_mask.jpg'
    // const imageElement = document.createElement('img');
    // imageElement.src = 'face_with_mask.jpg'; // Update this to the correct path
    // imageElement.style.width = '100%';
    // divElement.appendChild(imageElement);

    // Hide the submit button
    document.querySelector('input[type="submit"]').style.display = 'none';

    // Sort the results
    const sortedPredictions = Object.entries(predictions).sort((a, b) => b[1] - a[1]);

    // Create elements to display the results
    sortedPredictions.forEach(([key, value]) => {
      const pElement = document.createElement('p');
      pElement.textContent = `${key.charAt(0).toUpperCase() + key.slice(1)}: ${(value).toFixed(2)}%`;
      pElement.style.fontSize = '1.5em';
      pElement.style.color = key === 'real' ? 'green' : 'red';
      pElement.style.margin = '10px 0';
      pElement.style.padding = '10px';
      pElement.style.backgroundColor = '#333';
      pElement.style.borderRadius = '5px';
      divElement.appendChild(pElement);
    });

    // Hide the image preview
    document.querySelector('#preview').style.display = 'none';
  })
  .catch(error => console.error(error));
});
