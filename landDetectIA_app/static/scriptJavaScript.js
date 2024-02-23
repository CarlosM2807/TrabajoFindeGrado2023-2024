document.addEventListener("DOMContentLoaded", function () {

    // Variables de elementos del DOM
    const input = document.getElementById("input");
    const errorMensaje = document.getElementById("error-mensaje");
    const exitoMensaje = document.getElementById("exito-mensaje");
    const preview = document.getElementById("preview");
    const dropZone = document.getElementById("archivo"); 
    const texto = document.getElementById("drop_zone"); 
    const botonCancelar = document.getElementById("botoncancelar");
    const imagen_subida = document.getElementById("subida");

    // Permite cargar una imagen y realizar validaciones sobre la misma
    function cargarImagen(file) {

        //Extensiones permitidas
        const permitido = ["png", "jpg", "jpeg"];
        const extensionInput = file.name.split('.').pop();

        if (!permitido.includes(extensionInput.toLowerCase())) {

            // Extensión no permitida, mostrar mensaje de advertencia
            errorMensaje.style.display = "block";
            exitoMensaje.style.display = "none";
            preview.style.display = "none";
            botonCancelar.style.display = "none";
            imagen_subida.style.display = "block";

            // Limpiamos la carga
            input.value = "";
        } else {

            // Extensión permitida
            errorMensaje.style.display = "none";
            imagen_subida.style.display = "none";
            exitoMensaje.style.display = "block";
            botonCancelar.style.display = "block";

            // Asignar el archivo al input de tipo file
            var dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;

            // Disparar el evento change
            var event = new Event('change');
            input.dispatchEvent(event);

            // Mostrar previsualización de la imagen
            const reader = new FileReader();
            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = "block";
                texto.style.display = "none";
            };
            reader.readAsDataURL(file);
        }
    }

    // Evento para el cambio en el input
    input.addEventListener("change", function () {
        const file = input.files[0];
        cargarImagen(file);
    });

    // Evento para el envio del formulario
    const form = document.querySelector('form');
    form.addEventListener("submit", function (event) {
        const inputFile = document.getElementById('input');

        // NO se ha seleccionado imagen, entonces mostrar pop up
        if (inputFile.files.length === 0) {
            event.preventDefault();
            openPopup();
        }
    });

    // Permite que todo el cuadro sea zona de drag and drop
    window.onload = function () {
        var archivo = document.getElementById('archivo');
        var input = document.getElementById('input');

        archivo.ondrop = drop;
        archivo.ondragover = allowDrop;

        archivo.onclick = function () {
            input.click();
        };
    };

    // Funcion que permite manejar el evento de soltar archivos en el drag and drop
    function drop(event) {
        event.preventDefault();
        dropZone.classList.remove("drag-over");

        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            cargarImagen(file);
        }
    }

    // Eventos para arrastrar y soltar archivos 
    dropZone.addEventListener("dragover", allowDrop);
    dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("drag-over");
    });
    dropZone.addEventListener("drop", drop);


    // Funcion que permite arrastrar y soltar archivos
    function allowDrop(event) {
        event.preventDefault();
        dropZone.classList.add("drag-over");
    }

    // Función para abrir el pop up cuando no se carga una imagen
    function openPopup() {
        document.getElementById('noImagePopup').style.display = 'block';
    }

    // Función para cerrar el pop up cuando se clicka en el boton
    function closePopup() {
        document.getElementById('noImagePopup').style.display = 'none';
    }
});
