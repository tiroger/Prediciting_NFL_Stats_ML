// from data.js
var tableData = data;

// Creating table in HTML page

// Selecting table body element
var tableBody = d3.select("tbody");

// Designating column names for table
var tableColumns = ["datetime", "city", "state", "country", "shape", "durationMinutes", "comments"];

// Defining funcion to populate table
var renderTable = (dataInput) => {

    dataInput.forEach(ufo_sightings => {
        var row = tableBody.append("tr");
        tableColumns.forEach(column => row.append("td").text(ufo_sightings[column])
        )
    });
}

//Populate table
renderTable(tableData);

// Making table searchable/filterting data by date

// Creating variable for input date
var datetime = d3.select("#datetime");

// Adding an event listener to search button
var button = d3.select("#filter-btn");

button.on('click', () => {
   
    // Prevent refreshing
    event.preventDefault();
    
    var inputDate = datetime.property('value').trim();
    console.log(inputDate)

    var filteredData = data.filter(data => data.datetime === inputDate);
    console.log(filteredData);

    let response = {
        filteredData
    }

    if (response.filteredData.length !== 0) {
        tableBody.html("");
        renderTable(filteredData);
    }
    
    else {
        // Add filtered sighting to table
        tableBody.html("");
        tableBody.append("tr").append("td").text("No results found!");
    }

});


// Resetting input field after search
var resetButton = d3.select("#reset-btn");
    resetButton.on("click", () => {
    tableBody.html("");
    console.log("Table reset")
    renderTable(tableData);
    d3.select("#datetime").resetButton()
});

