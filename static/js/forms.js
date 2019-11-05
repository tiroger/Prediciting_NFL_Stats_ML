// forms.js
$(document).ready(function() {
  // process the form
  $("form").submit(function(event) {
    $(".form-group").removeClass("has-error"); // remove the error class
    $(".help-block").remove(); // remove the error text

    // get the form data
    // there are many ways to get this data using jQuery (you can use the class or id also)

    let formData = {
      yardline: $("#yardline").val(),
      secondsHalf: $("#secondsHalf").val(),
      secondsGame: $("#secondsGame").val(),
      down: $("#down").val(),
      ydstogo: $("#ydstogo").val(),
      scoreDif: $("#scoreDif").val(),
      humidity: $("#humidity").val()
    };

    // process the form
    $.ajax({
      type: "POST", // define the type of HTTP verb we want to use (POST for our form)
      url: "/results", // the url where we want to POST
      data: formData, // our data object
      dataType: "json", // what type of data do we expect back from the server
      encode: true
    })
      // using the done promise callback
      .done(function(data) {
        // log data to the console so we can see
        // console.log(data);

        // here we will handle errors and validation messages
        if (!data.success) {
          document.getElementById(
            "textOutput"
          ).innerHTML = `Based on these game and play conditions you should ${data}`;
          if (data === "RUN")
            $("#imageOutput").attr("src", "static/images/run.gif");
          else if (data === "PASS")
            $("#imageOutput").attr("src", "static/images/pass.gif");
          else if (data === "FIELD GOAL")
            $("#imageOutput").attr("src", "static/images/field_goal.gif");
          else $("#imageOutput").attr("src", "static/images/punt.gif");
        }
      })

      // using the fail promise callback
      .fail(function(data) {
        // show any errors
        // best to remove for production
        // console.log(data);
      });

    // stop the form from submitting the normal way and refreshing the page
    event.preventDefault();
  });
});
