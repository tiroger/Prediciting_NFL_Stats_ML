
// Creating canvas
var margin = {top: 80, right: 180, bottom: 80, left: 180},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

var svg = d3.select("#dropdown-viz").append("svg")
	.attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


// Load and munge data, then make the visualization.
var modelScores = ["Test Score"];
var url = "/scores";
// var promise = ;

// Getting data from table route
d3.json(url).then(function(data){
  console.log(data);
  var modelMap = {};
  data.forEach(function(d) {
    var model = d.model;
    console.log(model)
    var accuracy = d.accuracy
    console.log(accuracy)
    modelMap[model] = [];
    modelScores.forEach(function(score) {
      modelMap[model].push(+d[score]);
      // console.log(modelMap)
    });
  })
// makeVis(modelMap);


	// create the drop down menu of models
	var selector = d3.select(".dropdown")
		.append("select")
		.attr("id", "model")
		.selectAll("option")
		.data(data)
		.enter().append("option")
		.text(function(d) { return d.model; })
		.attr("value", function (d, i) {
			return i;
		});



  });

  // generate a random index value and set the selector to the city
// at that index value in the data array
// var index = Math.round(Math.random() * data.length);
// d3.select("#model").property("selectedIndex", index);
//
// // append a paragraph tag to the body that shows the city name and it's population
// d3.select("body")
//       .append("p")
//       .data(data)
//       .text(function(d){
//         return data[index]['train_score'] + data[index]['test_score'];
//       })
//
// // when the user selects a city, set the value of the index variable
// // and call the update(); function
// d3.select("#model")
// .on("change", function(d) {
//   index = this.value;
//   update();
// })
//
// // update the paragraph text to match the selection made by the user
// function update(){
//   d3.selectAll("p")
//     .data(data)
//     .text(function(d){
//       return data[index]['train_score'] + data[index]['test_score'];
//     })
// };
