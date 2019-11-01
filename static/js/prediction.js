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
  // let modelValue = [
  //   [
  //     inputYardline,
  //     inputSecondsHalf,
  //     inputSecondsGame,
  //     inputDown,
  //     inputYdstogo,
  //     inputScoreDif
  //   ]
  // ];

  // d3.json(modelValue, function ())

  // d3.request("/result").send("POST", modelValue, function(modelValue) {
  //   console.log(modelValue);
  // });

  // var modelValue = [[19.0, 3, 1.0, 10, 3.0, 3.0, 35, 74]];
  // var modelValue = modelValue;

  // $.ajax({
  //   type: "POST",
  //   url: "/predictions",
  //   data: modelValue
  // });
  // console.log(modelValue);
  // let predictionOutput = d3.select("#predictionOutput");
  let output = d3.select("#output");
  let random = Math.floor(Math.random() * 2);
  console.log(random);

  if (random < 1) {
    output.attr("src", "static/images/run.gif");
  } else {
    output.attr("src", "static/images/pass.gif");
  }

  // if (ydstogo < 5) {
  //   output.attr("src", "static/images/run.gif");
  // } else if (down > 3) {
  //   // Add filtered sighting to table
  //   output.attr("src", "static/images/pass.gif");
  // }

  return modelValue;
}

submitbutton.on("click", submitData);

// Source for addional follow up work below

// magic.js
// $(document).ready(function() {
//   // process the form
//   $("form").submit(function(event) {
//     // get the form data
//     // there are many ways to get this data using jQuery (you can use the class or id also)
//     var formData = {
//       yardline: $("input[yardline=yardline]").val(),
//       down: $("input[name=down]").val(),
//       ydstogo: $("input[name=ydstogo]").val(),
//       scoredif: $("input[name=scoreDif]").val(),
//       weather1: $("input[name=weather1]").val(),
//       weather2: $("input[name=weather2]").val(),
//       secondsHalf: $("input[name=secondsHalf]").val(),
//       secondsGame: $("input[name=secondsGame]").val()
//     };

//     // process the form
//     $.ajax({
//       type: "POST", // define the type of HTTP verb we want to use (POST for our form)
//       url: "/result", // the url where we want to POST
//       data: formData, // our data object
//       dataType: "json", // what type of data do we expect back from the server
//       encode: true
//     })
//       // using the done promise callback
//       .done(function(data) {
//         // log data to the console so we can see
//         console.log(data);

//         // here we will handle errors and validation messages
//       });

//     // stop the form from submitting the normal way and refreshing the page
//     event.preventDefault();
//   });
// });
