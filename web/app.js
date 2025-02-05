const form = document.getElementById("uploadForm");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    if (!file) {
        resultDiv.innerHTML = "<p>Por favor, selecciona una imagen.</p>";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/classify/", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            resultDiv.innerHTML = `
                <h3>Resultado</h3>
                <p><strong>Clase predicha:</strong> ${data.predicted_label}</p>
                <img src="data:image/jpeg;base64,${data.original_image}" alt="Imagen original">
            `;
        } else {
            resultDiv.innerHTML = "<p>Error al procesar la imagen.</p>";
        }
    } catch (error) {
        console.error("Error de conexión:", error);
        resultDiv.innerHTML = "<p>Error de conexión con el servidor.</p>";
    }
});
