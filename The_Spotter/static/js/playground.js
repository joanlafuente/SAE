


document.addEventListener("DOMContentLoaded", function () {
  const openBtn = document.getElementById("openbtn");
  const closeBtn = document.getElementById("closebtn");
  const sidebar = document.getElementById("mySidebar");
  const main = document.getElementById("main");
  const centered = document.querySelector(".centered");

  openBtn.addEventListener("click", () => {
    sidebar.style.width = "250px";
    main.style.marginLeft = "250px";
    openBtn.style.display = "none"; // Hide the Open Settings button when the sidebar is opened
  });

  closeBtn.addEventListener("click", () => {
    sidebar.style.width = "0";
    main.style.marginLeft = "0";
    openBtn.style.display = "block"; // Show the Open Settings button when the sidebar is closed
  });
});



function displayGraph() {
  var iframe = document.getElementById("graphFrame");
  iframe.src = "/playground_graph1"; // Route to serve the first graph
  iframe.style.display = "block";
  document.getElementById('closeGraph1').style.display = "block"; // Show the close button
}

function displayGraph2() {
  var iframe = document.getElementById('graphFrame2');
  iframe.src = "/playground_graph2"; // Route to serve the second graph
  iframe.style.display = "block";
  document.getElementById('closeGraph2').style.display = "block"; // Show the close button
}




document.getElementById('closeGraph1').addEventListener('click', function() {
  document.getElementById('graphFrame').style.display = 'none';
  this.style.display = 'none';
});

document.getElementById('closeGraph2').addEventListener('click', function() {
  document.getElementById('graphFrame2').style.display = 'none';
  this.style.display = 'none';
});





