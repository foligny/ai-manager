// Charts Module
class ChartsManager {
    constructor() {
        this.charts = new Map();
    }

    renderCharts(metrics, runId) {
        const chartsContainer = document.getElementById('charts-container');
        if (!chartsContainer) return;

        // Clear existing charts
        chartsContainer.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

        // Group metrics by name
        const metricsByType = this.groupMetricsByType(metrics);

        // Create charts for each metric type
        Object.keys(metricsByType).forEach(metricName => {
            const metricData = metricsByType[metricName];
            this.createChart(metricName, metricData, runId, chartsContainer);
        });
    }

    groupMetricsByType(metrics) {
        const grouped = {};
        metrics.forEach(metric => {
            const metricName = metric.metric_name || metric.name;
            if (!grouped[metricName]) {
                grouped[metricName] = [];
            }
            grouped[metricName].push(metric);
        });
        return grouped;
    }

    createChart(metricName, metricData, runId, container) {
        // Create chart container
        const chartDiv = document.createElement('div');
        chartDiv.className = 'col-md-6 mb-4';
        chartDiv.id = `chart-${metricName}-${runId}`;
        
        const chartTitle = document.createElement('h5');
        chartTitle.textContent = metricName.replace(/_/g, ' ').toUpperCase();
        chartDiv.appendChild(chartTitle);
        
        container.appendChild(chartDiv);

        // Small delay to ensure container is properly sized
        setTimeout(() => {
            // Format x-axis values properly
            const xValues = metricData.map((m, index) => {
                // Use step number directly, or index if step is not available
                return m.step || index;
            });

            // Create Plotly chart with better formatting
            const trace = {
                x: xValues,
                y: metricData.map(m => m.value),
                type: 'scatter',
                mode: 'lines+markers',
                name: `Run ${runId}`,
                line: { color: '#6366f1', width: 2 },
                marker: { size: 6, color: '#6366f1' }
            };

            const layout = {
                title: {
                    text: metricName.replace(/_/g, ' ').toUpperCase(),
                    font: { size: 16, color: '#f9fafb' }
                },
                xaxis: { 
                    title: { text: 'Step', font: { color: '#f9fafb' } },
                    tickfont: { color: '#f9fafb' },
                    gridcolor: '#4b5563',
                    zerolinecolor: '#4b5563',
                    type: 'linear'
                },
                yaxis: { 
                    title: { text: metricName.replace(/_/g, ' ').toUpperCase(), font: { color: '#f9fafb' } },
                    tickfont: { color: '#f9fafb' },
                    gridcolor: '#4b5563',
                    zerolinecolor: '#4b5563',
                    type: 'linear'
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#f9fafb' },
                margin: { t: 60, r: 20, b: 60, l: 80 },
                hovermode: 'x unified',
                showlegend: false,
                height: 350,
                autosize: true
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                displaylogo: false,
                useResizeHandler: true
            };

            Plotly.newPlot(chartDiv.id, [trace], layout, config);
            
            // Store chart reference
            this.charts.set(`${metricName}-${runId}`, chartDiv.id);
        }, 100);
    }

    updateChart(data) {
        const { run_id, metric_name, name, value, step } = data;
        const metricName = metric_name || name;
        const chartId = `${metricName}-${run_id}`;
        
        if (this.charts.has(chartId)) {
            const elementId = this.charts.get(chartId);
            const chartDiv = document.getElementById(elementId);
            
            if (chartDiv) {
                Plotly.extendTraces(elementId, {
                    x: [[step]],
                    y: [[value]]
                });
            }
        }
    }

    clearCharts() {
        this.charts.clear();
        const chartsContainer = document.getElementById('charts-container');
        if (chartsContainer) {
            chartsContainer.innerHTML = '';
        }
    }
}

// Global charts manager instance
window.chartsManager = new ChartsManager(); 