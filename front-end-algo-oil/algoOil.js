function toggleGraph(graphId) {
    var graph = document.getElementById(graphId);
    var isVisible = graph.style.display === 'block';

    

    graph.style.display = isVisible ? 'none' : 'block';
    graph.classList.toggle('enlarged', true);

}

document.querySelectorAll('.graph-button').forEach(button => {
    button.addEventListener('click', function () {
        document.querySelectorAll('.graph-container.enlarged').forEach(graph => {
            graph.classList.remove('enlarged');
            graph.style.display = 'none';

        });
        var graphID = this.getAttribute('data-graph');
        var originalData = this.getAttribute('original-data');
        var predictionData = this.getAttribute('prediction-data');
        toggleGraph(graphID);
        loadAndPlotData(originalData, predictionData, graphID);
    });
});

document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
        document.querySelectorAll('.graph-container.enlarged').forEach(graph => {
            graph.classList.remove('enlarged');
            graph.style.display = 'none';
            
        });
        
        document.body.style.overflow = 'auto';
    }
});

function loadAndPlotData(original, prediction, graphId) {
    Plotly.d3.csv(original, function (err, originalRows) {
        Plotly.d3.csv(prediction, function (err, predictionRows) {
            function unpack(rows, key) {
                return rows.map(function (row) { return row[key]; });
            }

            var originalTrace = {
                type: "scatter",
                mode: "lines",
                name: 'OriginalData',
                x: unpack(originalRows, 'ds'),
                y: unpack(originalRows, 'y'),
                line: { color: '#1f77b4' } // blue
            };

            var predictionTrace = {
                type: "scatter",
                mode: "lines",
                name: 'Prediction',
                x: unpack(predictionRows, 'ds'),
                y: unpack(predictionRows, 'yhat'),
                line: { color: '#ff7f0e' } // red
            };

            var data = [originalTrace, predictionTrace];

            var layout = {
                title: 'Predicted price',
                xaxis: {
                    title: 'Date'
                },
                yaxis: {
                    title: 'Value'
                }
            };

            Plotly.newPlot(graphId, data, layout);
        });
    });
}
