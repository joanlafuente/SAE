
document.addEventListener("DOMContentLoaded", function () {
  const openBtn = document.getElementById("openbtn");
  const closeBtn = document.getElementById("closebtn");
  const sidebar = document.getElementById("mySidebar");
  const main = document.getElementById("main");

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
  var container1 = document.getElementById("graph-container1");
  var container2 = document.getElementById("graph-container2");

  // Only one graph can be open at a time
  if (container1.style.display === "block" || container2.style.display === "block") {
    alert("Only one graph can be uploaded at a time.");
  } else {
    var iframe1 = document.getElementById("graphFrame");
    iframe1.style.display = "block";
    iframe1.src = "/playground_graph1"; // Route to serve the first graph
    document.getElementById('closeGraph1').style.display = "block"; // Show the close button
    container1.style.display = "block"; // Show the container
  }
}

function displayGraph2() {
  var container1 = document.getElementById("graph-container1");
  var container2 = document.getElementById("graph-container2");

  if (container1.style.display === "block" || container2.style.display === "block") {
    alert("Only one graph can be uploaded at a time.");
  } else {
    var iframe2 = document.getElementById("graphFrame2");
    iframe2.style.display = "block";
    iframe2.src = "/playground_graph2"; // Route to serve the second graph
    document.getElementById('closeGraph2').style.display = "block"; // Show the close button
    container2.style.display = "block"; // Show the container
  }
}

document.getElementById('closeGraph1').addEventListener('click', function() {
  var container1 = document.getElementById("graph-container1");
  document.getElementById('graphFrame').style.display = 'none';
  this.style.display = 'none';
  container1.style.display = 'none';
});

document.getElementById('closeGraph2').addEventListener('click', function() {
  var container2 = document.getElementById("graph-container2");
  document.getElementById('graphFrame2').style.display = 'none';
  this.style.display = 'none';
  container2.style.display = 'none';
});

function Filter1() {
  var iframe1 = document.getElementById("graphFrame");
  var iframe2 = document.getElementById("graphFrame2");
  
  if (iframe1.style.display === "none" && iframe2.style.display === "none") {
    alert("A graph needs to be uploaded.");
  } 
  if (iframe2.style.display === "block") {
    iframe2.src = "/playground_graph2/Supervised"; // Route to serve the second graph
    iframe2.style.display = "block";
    document.getElementById('closeGraph2').style.display = "block"; 
  }
  if (iframe1.style.display === "block") {
    iframe1.src = "/playground_graph1/Supervised"; // Route to serve the first graph
    iframe1.style.display = "block";
    document.getElementById('closeGraph1').style.display = "block"; 
  }
}

function Filter2() {
  var iframe1 = document.getElementById("graphFrame");
  var iframe2 = document.getElementById("graphFrame2");
  
  if (iframe1.style.display === "none" && iframe2.style.display === "none") {
    alert("A graph needs to be uploaded.");
  } 
  if (iframe2.style.display === "block") {
    iframe2.src = "/playground_graph2/Autoencoder"; // Route to serve the second graph
    iframe2.style.display = "block";
    document.getElementById('closeGraph2').style.display = "block"; 
  }
  if (iframe1.style.display === "block") {
    iframe1.src = "/playground_graph1/Autoencoder"; // Route to serve the first graph
    iframe1.style.display = "block";
    document.getElementById('closeGraph1').style.display = "block"; 
  }
}

function Filter3() {
  var iframe1 = document.getElementById("graphFrame");
  var iframe2 = document.getElementById("graphFrame2");
  
  if (iframe1.style.display === "none" && iframe2.style.display === "none") {
    alert("A graph needs to be uploaded.");
  } 
  if (iframe2.style.display === "block") {
    alert("SelfSupervised is not available for the Yelp graph");
  }
  if (iframe1.style.display === "block") {
    iframe1.src = "/playground_graph1/SelfSupervised"; // Route to serve the first graph
    iframe1.style.display = "block";
    document.getElementById('closeGraph1').style.display = "block"; 
  }
}