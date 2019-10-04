// Creating variable for inputs

const yardline = d3.select("#yardline");
const secondsHalf = d3.select("#secondsHalf");
const secondsGame = d3.select("#secondsGame");
const down = d3.select("#down");
const ydstogo = d3.select("#ydstogo");
const scoreDif = d3.select("#scoreDif");

// Adding an event listener to search button
const submitbutton = d3.select("#submit-data-btn");

function submitData() {
  let inputYardline = yardline.property("value").trim();
  let inputSecondsHalf = secondsHalf.property("value").trim();
  let inputSecondsGame = secondsGame.property("value").trim();
  let inputDown = down.property("value").trim();
  let inputYdstogo = ydstogo.property("value").trim();
  let inputScoreDif = scoreDif.property("value").trim();
  let modelValue = [
    [
      inputYardline,
      inputSecondsHalf,
      inputSecondsGame,
      inputDown,
      inputYdstogo,
      inputScoreDif
    ]
  ];

  console.log(modelValue);
}

submitbutton.on("click", submitData);

let response = "pass";
// let predictionOutput = d3.select("#predictionOutput");
let output = d3.select("#output");
if (response == "pass") {
  output.attr("src", "static/images/pass.gif");
} else if (response == "run") {
  // Add filtered sighting to table
  output.attr("src", "static/images/run.gif");
}

// Resetting input field after search
var resetButton = d3.select("#reset-btn");
resetButton.on("click", () => {
  tableBody.html("");
  console.log("Table reset");
  renderTable(tableData);
  d3.select("#datetime").resetButton();
});
