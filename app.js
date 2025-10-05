function getRecommendations() {
  const userId = document.getElementById("userId").value;
  const movieTitle = document.getElementById("movieTitle").value;

  fetch(`http://localhost:5000/recommend?user_id=${userId}&movie_title=${encodeURIComponent(movieTitle)}`)
    .then(response => response.json())
    .then(data => {
      const list = document.getElementById("recommendations");
      list.innerHTML = "";
      data.recommendations.forEach(movie => {
        const li = document.createElement("li");
        li.textContent = movie;
        list.appendChild(li);
      });
    });
}
