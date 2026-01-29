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
        <span>${file.name}</span>
    `;

    item.onclick = () => {
        mainVideo.src = videoURL;
        mainVideo.play();
    };

    videoList.appendChild(item);

    // Auto play first uploaded video
    if (!mainVideo.src) {
        mainVideo.src = videoURL;
        mainVideo.play();
    }
});
