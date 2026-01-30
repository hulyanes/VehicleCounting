const uploadInput = document.getElementById("upload");
const videoList = document.getElementById("videoList");
const mainVideo = document.getElementById("mainVideo");

uploadInput.addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;

    const videoURL = URL.createObjectURL(file);

    // Create list item
    const item = document.createElement("div");
    item.className = "video-item";

    item.innerHTML = `
        <div class="thumb"></div>
        <span class="video-name">${file.name}</span>
        <button class="delete-btn"><i class="bi bi-trash-fill"></i></button>
    `;

    // Play video when clicking item (except delete)
    item.addEventListener("click", (e) => {
        if (e.target.classList.contains("delete-btn")) return;
        mainVideo.src = videoURL;
        mainVideo.play();
    });

    // Delete button logic
    item.querySelector(".delete-btn").addEventListener("click", (e) => {
        e.stopPropagation();

        // If deleting currently playing video
        if (mainVideo.src === videoURL) {
            mainVideo.pause();
            mainVideo.src = "";
        }

        URL.revokeObjectURL(videoURL);
        item.remove();
    });

    videoList.appendChild(item);

    // Auto play first uploaded video
    if (!mainVideo.src) {
        mainVideo.src = videoURL;
        mainVideo.play();
    }

    // Allow re-uploading same file
    uploadInput.value = "";
});
