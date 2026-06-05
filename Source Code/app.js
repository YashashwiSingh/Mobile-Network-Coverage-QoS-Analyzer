console.log('Starting application...');

try {
    const { useState, useEffect, useRef } = React;
    console.log('React hooks loaded successfully');

    // Simple data generator
    const generateSampleData = () => {
        console.log('Generating sample data...');
        const localities = ['Downtown', 'Suburbs', 'Airport', 'Mall', 'University', 'Hospital', 'Park', 'Stadium'];
        const networkTypes = ['4G', '5G', 'LTE', '3G'];
        
        const data = Array.from({ length: 50 }, (_, i) => ({
            timestamp: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
            locality: localities[Math.floor(Math.random() * localities.length)],
            latitude: 12.9716 + (Math.random() - 0.5) * 0.1,
            longitude: 77.5946 + (Math.random() - 0.5) * 0.1,
            signalStrength: -120 + Math.random() * 70,
            signalQuality: Math.random() * 100,
            dataThroughput: Math.random() * 100,
            latency: 10 + Math.random() * 200,
            networkType: networkTypes[Math.floor(Math.random() * networkTypes.length)],
            bb60cMeasurement: -120 + Math.random() * 70,
            srsranMeasurement: -120 + Math.random() * 70,
            bladeRFMeasurement: -120 + Math.random() * 70
        }));
        
        console.log('Generated', data.length, 'sample records');
        return data;
    };

    // Simple ML predictor
    class SimpleMLPredictor {
        constructor() {
            this.coefficients = null;
            this.isTrained = false;
            console.log('ML Predictor initialized');
        }

        train(data) {
            try {
                console.log('Training ML model with', data.length, 'samples');
                
                if (data.length < 10) {
                    console.log('Not enough data for training');
                    return;
                }
                
                // Simple linear regression
                const features = data.map(d => [
                    1, // bias
                    d.signalStrength / 100,
                    d.latitude,
                    d.longitude,
                    d.networkType === '5G' ? 1 : d.networkType === '4G' ? 0.8 : d.networkType === 'LTE' ? 0.6 : 0.4,
                    d.dataThroughput / 100,
                    d.latency / 1000
                ]);
                
                const targets = data.map(d => d.signalQuality / 100);
                
                // Calculate coefficients using normal equation
                this.coefficients = this.calculateLinearRegression(features, targets);
                this.isTrained = true;
                console.log('ML model trained successfully');
            } catch (error) {
                console.error('ML training failed:', error);
            }
        }

        calculateLinearRegression(X, y) {
            try {
                const Xt = this.transpose(X);
                const XtX = this.matrixMultiply(Xt, X);
                const XtXInv = this.matrixInverse(XtX);
                const XtY = this.matrixVectorMultiply(Xt, y);
                return this.matrixVectorMultiply(XtXInv, XtY);
            } catch (error) {
                console.error('Linear regression calculation failed:', error);
                return [0, 0, 0, 0, 0, 0, 0]; // fallback
            }
        }

        transpose(matrix) {
            return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
        }

        matrixMultiply(a, b) {
            const result = [];
            for (let i = 0; i < a.length; i++) {
                result[i] = [];
                for (let j = 0; j < b[0].length; j++) {
                    let sum = 0;
                    for (let k = 0; k < b.length; k++) {
                        sum += a[i][k] * b[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        }

        matrixVectorMultiply(matrix, vector) {
            return matrix.map(row => 
                row.reduce((sum, val, i) => sum + val * vector[i], 0)
            );
        }

        matrixInverse(matrix) {
            const n = matrix.length;
            const identity = Array(n).fill().map((_, i) => 
                Array(n).fill().map((_, j) => i === j ? 1 : 0)
            );
            
            const augmented = matrix.map((row, i) => [...row, ...identity[i]]);
            
            for (let i = 0; i < n; i++) {
                let maxRow = i;
                for (let k = i + 1; k < n; k++) {
                    if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                        maxRow = k;
                    }
                }
                [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
                
                const pivot = augmented[i][i];
                if (Math.abs(pivot) < 1e-10) {
                    console.warn('Matrix is singular, using fallback');
                    return identity;
                }
                
                for (let k = 0; k < 2 * n; k++) {
                    augmented[i][k] /= pivot;
                }
                
                for (let j = 0; j < n; j++) {
                    if (i !== j) {
                        const factor = augmented[j][i];
                        for (let k = 0; k < 2 * n; k++) {
                            augmented[j][k] -= factor * augmented[i][k];
                        }
                    }
                }
            }
            
            return augmented.map(row => row.slice(n));
        }

        predict(signalStrength, latitude, longitude, networkType, dataThroughput = 0, latency = 0) {
            if (!this.isTrained || !this.coefficients) return null;

            try {
                const features = [
                    1, // bias
                    signalStrength / 100,
                    latitude,
                    longitude,
                    networkType === '5G' ? 1 : networkType === '4G' ? 0.8 : networkType === 'LTE' ? 0.6 : 0.4,
                    dataThroughput / 100,
                    latency / 1000
                ];
                
                const prediction = this.coefficients.reduce((sum, coef, i) => 
                    sum + coef * features[i], 0
                );
                
                return Math.max(0, Math.min(100, prediction * 100));
            } catch (error) {
                console.error('Prediction failed:', error);
                return null;
            }
        }
    }

    const NetworkAnalyzer = () => {
        console.log('NetworkAnalyzer component initializing...');
        
        const [data, setData] = useState([]);
        const [filteredData, setFilteredData] = useState([]);
        const [filters, setFilters] = useState({
            locality: '',
            networkType: '',
            minSignalStrength: '',
            maxLatency: ''
        });
        const [loading, setLoading] = useState(false);
        const [error, setError] = useState('');
        const [debugInfo, setDebugInfo] = useState('Application loaded successfully');
        const [mlPredictor, setMlPredictor] = useState(new SimpleMLPredictor());
        const [predictions, setPredictions] = useState([]);

        console.log('State initialized');

        // Parse CSV data
        const parseCSV = (csvText) => {
            try {
                console.log('Parsing CSV data...');
                const lines = csvText.split('\n').filter(line => line.trim());
                console.log('CSV lines:', lines.length);
                
                if (lines.length < 2) {
                    throw new Error('CSV must have header and data');
                }

                const data = [];
                for (let i = 1; i < lines.length; i++) {
                    const values = lines[i].split(',').map(v => v.trim());
                    if (values.length >= 9) {
                        data.push({
                            timestamp: values[0] || '',
                            locality: values[1] || '',
                            latitude: parseFloat(values[2]) || 0,
                            longitude: parseFloat(values[3]) || 0,
                            signalStrength: parseFloat(values[4]) || 0,
                            signalQuality: parseFloat(values[5]) || 0,
                            dataThroughput: parseFloat(values[6]) || 0,
                            latency: parseFloat(values[7]) || 0,
                            networkType: values[8] || '',
                            bb60cMeasurement: parseFloat(values[9]) || 0,
                            srsranMeasurement: parseFloat(values[10]) || 0,
                            bladeRFMeasurement: parseFloat(values[11]) || 0
                        });
                    }
                }
                console.log('Parsed', data.length, 'records from CSV');
                return data;
            } catch (err) {
                console.error('CSV parsing error:', err);
                throw new Error('CSV parsing failed: ' + err.message);
            }
        };

        // Handle file upload
        const handleFileUpload = (event) => {
            console.log('File upload triggered');
            const file = event.target.files[0];
            if (!file) {
                console.log('No file selected');
                return;
            }

            console.log('File selected:', file.name, file.type);
            setLoading(true);
            setError('');
            setDebugInfo('Starting file upload...');

            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    console.log('File read successfully, parsing CSV...');
                    setDebugInfo('File read successfully, parsing CSV...');
                    
                    const parsedData = parseCSV(e.target.result);
                    console.log('CSV parsed successfully:', parsedData.length, 'records');
                    
                    setDebugInfo(`Parsed ${parsedData.length} records successfully`);
                    setData(parsedData);
                    // Also update filtered data immediately to render charts without waiting
                    setFilteredData(parsedData);
                    
                    // Train ML model
                    console.log('Training ML model...');
                    mlPredictor.train(parsedData);
                    setDebugInfo('ML model trained successfully');
                    
                    console.log('File upload completed successfully');
                } catch (err) {
                    console.error('File upload error:', err);
                    setError('Error: ' + err.message);
                    setDebugInfo('Error: ' + err.message);
                }
                setLoading(false);
                // Ensure charts render after DOM updates
                setTimeout(() => {
                    createCharts();
                }, 150);
            };
            
            reader.onerror = (error) => {
                console.error('File reader error:', error);
                setError('Error reading file');
                setDebugInfo('Error reading file');
                setLoading(false);
            };
            
            reader.readAsText(file);
        };

        // Load sample data
        const loadSampleData = () => {
            console.log('Loading sample data...');
            setLoading(true);
            setError('');
            setDebugInfo('Loading sample data...');
            
            try {
                const sampleData = generateSampleData();
                console.log('Sample data generated:', sampleData.length, 'records');
                
                setData(sampleData);
                // Also set filtered data immediately so charts can render
                setFilteredData(sampleData);
                setDebugInfo(`Loaded ${sampleData.length} sample records`);
                
                // Train ML model
                console.log('Training ML model with sample data...');
                mlPredictor.train(sampleData);
                setDebugInfo('ML model trained successfully');
                
                console.log('Sample data loading completed');
            } catch (error) {
                console.error('Sample data loading error:', error);
                setError('Error loading sample data: ' + error.message);
                setDebugInfo('Error loading sample data: ' + error.message);
            }
            
            setLoading(false);
            // Ensure charts render after state settles
            setTimeout(() => {
                createCharts();
            }, 150);
        };

        // Apply filters
        const applyFilters = () => {
            console.log('Applying filters...');
            let filtered = [...data];
            
            if (filters.locality) {
                filtered = filtered.filter(d => d.locality === filters.locality);
            }
            if (filters.networkType) {
                filtered = filtered.filter(d => d.networkType === filters.networkType);
            }
            if (filters.minSignalStrength) {
                filtered = filtered.filter(d => d.signalStrength >= parseFloat(filters.minSignalStrength));
            }
            if (filters.maxLatency) {
                filtered = filtered.filter(d => d.latency <= parseFloat(filters.maxLatency));
            }
            
            console.log('Filtered data:', filtered.length, 'records');
            setFilteredData(filtered);
        };

        // Generate ML predictions
        const generatePredictions = () => {
            console.log('Generating ML predictions...');
            if (!mlPredictor.isTrained || filteredData.length === 0) {
                console.log('Cannot generate predictions: Model trained =', mlPredictor.isTrained, 'Data length =', filteredData.length);
                setDebugInfo('Cannot generate predictions: Model not trained or no data');
                return;
            }

            try {
                const newPredictions = filteredData.slice(0, 10).map(point => {
                    const predictedQuality = mlPredictor.predict(
                        point.signalStrength,
                        point.latitude,
                        point.longitude,
                        point.networkType,
                        point.dataThroughput,
                        point.latency
                    );
                    
                    const accuracy = predictedQuality ? 
                        Math.max(0, 100 - Math.abs(predictedQuality - point.signalQuality)) : 0;
                    
                    return {
                        locality: point.locality,
                        actualQuality: point.signalQuality,
                        predictedQuality: predictedQuality || 0,
                        accuracy: accuracy,
                        networkType: point.networkType
                    };
                });

                console.log('Generated predictions:', newPredictions.length);
                setPredictions(newPredictions);
                setDebugInfo(`Generated ${newPredictions.length} ML predictions`);
            } catch (error) {
                console.error('Prediction generation error:', error);
                setError('Error generating predictions: ' + error.message);
            }
        };

        // Create charts with proper timing
        const createCharts = () => {
            console.log('Creating charts...');
            if (filteredData.length === 0) {
                console.log('No data for charts');
                return;
            }

            // Use setTimeout to ensure DOM is ready
            setTimeout(() => {
                try {
                    console.log('Chart.js available:', typeof Chart !== 'undefined');
                    
                    // Signal Strength Chart
                    const signalCanvas = document.getElementById('signalChart');
                    console.log('Signal canvas found:', !!signalCanvas);
                    
                    if (signalCanvas && typeof Chart !== 'undefined') {
                        const ctx = signalCanvas.getContext('2d');
                        
                        // Destroy existing chart safely
                        if (window.signalChart && typeof window.signalChart.destroy === 'function') {
                            window.signalChart.destroy();
                        }
                        
                        const chartData = filteredData.slice(0, 15);
                        console.log('Chart data for signal:', chartData.length, 'points');
                        
                        window.signalChart = new Chart(ctx, {
                            type: 'bar',
                            data: {
                                labels: chartData.map(d => d.locality),
                                datasets: [{
                                    label: 'Signal Strength (dBm)',
                                    data: chartData.map(d => d.signalStrength),
                                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                                    borderColor: 'rgba(102, 126, 234, 1)',
                                    borderWidth: 2,
                                    borderRadius: 8
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        display: true,
                                        position: 'top'
                                    }
                                },
                                scales: {
                                    y: {
                                        beginAtZero: false,
                                        title: {
                                            display: true,
                                            text: 'Signal Strength (dBm)'
                                        }
                                    }
                                }
                            }
                        });
                        console.log('Signal chart created successfully');
                    } else {
                        console.error('Signal canvas not found or Chart.js not available');
                    }

                    // Quality vs Throughput Scatter Chart
                    const scatterCanvas = document.getElementById('scatterChart');
                    console.log('Scatter canvas found:', !!scatterCanvas);
                    
                    if (scatterCanvas && typeof Chart !== 'undefined') {
                        const ctx2 = scatterCanvas.getContext('2d');
                        
                        // Destroy existing chart safely
                        if (window.scatterChart && typeof window.scatterChart.destroy === 'function') {
                            window.scatterChart.destroy();
                        }
                        
                        const scatterData = filteredData.slice(0, 50);
                        console.log('Chart data for scatter:', scatterData.length, 'points');
                        
                        window.scatterChart = new Chart(ctx2, {
                            type: 'scatter',
                            data: {
                                datasets: [{
                                    label: 'Quality vs Throughput',
                                    data: scatterData.map(d => ({
                                        x: d.signalQuality,
                                        y: d.dataThroughput
                                    })),
                                    backgroundColor: 'rgba(118, 75, 162, 0.6)',
                                    borderColor: 'rgba(118, 75, 162, 1)',
                                    borderWidth: 2,
                                    pointRadius: 6
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        display: true,
                                        position: 'top'
                                    }
                                },
                                scales: {
                                    x: {
                                        title: {
                                            display: true,
                                            text: 'Signal Quality (%)'
                                        },
                                        min: 0,
                                        max: 100
                                    },
                                    y: {
                                        title: {
                                            display: true,
                                            text: 'Data Throughput (Mbps)'
                                        },
                                        beginAtZero: true
                                    }
                                }
                            }
                        });
                        console.log('Scatter chart created successfully');
                    } else {
                        console.error('Scatter canvas not found or Chart.js not available');
                    }
                } catch (error) {
                    console.error('Chart creation error:', error);
                    setError('Error creating charts: ' + error.message);
                }
            }, 100); // Small delay to ensure DOM is ready
        };

        // Update filters when data or filters change
        useEffect(() => {
            console.log('useEffect triggered - data or filters changed');
            if (data.length > 0) {
                console.log('Applying filters with', data.length, 'records');
                applyFilters();
            }
        }, [data, filters]);

        // Generate predictions and charts when filtered data changes
        useEffect(() => {
            console.log('useEffect triggered - filtered data changed');
            if (filteredData.length > 0) {
                console.log('Creating charts and predictions with', filteredData.length, 'records');
                generatePredictions();
                createCharts();
            }
        }, [filteredData]);

        // Calculate statistics
        const calculateStats = () => {
            if (!filteredData.length) return {};
            
            const stats = {
                avgSignalStrength: (filteredData.reduce((sum, d) => sum + d.signalStrength, 0) / filteredData.length).toFixed(1),
                avgThroughput: (filteredData.reduce((sum, d) => sum + d.dataThroughput, 0) / filteredData.length).toFixed(1),
                avgLatency: (filteredData.reduce((sum, d) => sum + d.latency, 0) / filteredData.length).toFixed(1),
                avgQuality: (filteredData.reduce((sum, d) => sum + d.signalQuality, 0) / filteredData.length).toFixed(1),
                totalPoints: filteredData.length,
                excellentPoints: filteredData.filter(d => d.signalQuality >= 80).length,
                poorPoints: filteredData.filter(d => d.signalQuality < 40).length
            };
            
            console.log('Calculated stats:', stats);
            return stats;
        };

        const stats = calculateStats();
        const localities = [...new Set(data.map(d => d.locality))].filter(Boolean);
        const networkTypes = [...new Set(data.map(d => d.networkType))].filter(Boolean);

        console.log('Rendering component with data:', data.length, 'filtered:', filteredData.length);

        return (
            <div className="app-container">
                <div className="header">
                    <h1>📡 Mobile Network Coverage & QoS Analyzer</h1>
                    <p>Advanced analytics with ML predictions and intelligent recommendations</p>
                </div>

                <div className="controls-section">
                    <div className="file-upload">
                        <div className="upload-area">
                            <h3>📂 Upload Your Network Data</h3>
                            <p>Choose a CSV file with network measurement data</p>
                            <input 
                                type="file" 
                                accept=".csv" 
                                onChange={handleFileUpload}
                                style={{margin: '10px 0'}}
                            />
                        </div>
                        <div>
                            <button className="sample-data-btn" onClick={loadSampleData}>
                                🚀 Load Sample Data
                            </button>
                            <button className="sample-data-btn debug-btn" onClick={() => {
                                const debug = `Data: ${data.length}, Filtered: ${filteredData.length}, ML Trained: ${mlPredictor.isTrained}, Predictions: ${predictions.length}`;
                                setDebugInfo(debug);
                                console.log(debug);
                            }}>
                                🔍 Debug
                            </button>
                            <button className="sample-data-btn ml-btn" onClick={generatePredictions}>
                                🧠 Generate ML Predictions
                            </button>
                            <button className="sample-data-btn" onClick={() => {
                                console.log('Manually refreshing charts...');
                                createCharts();
                                setDebugInfo('Charts refreshed manually');
                            }} style={{background: '#17a2b8'}}>
                                📊 Refresh Charts
                            </button>
                        </div>
                    </div>

                    {data.length > 0 && (
                        <div className="filters">
                            <div className="filter-group">
                                <label>Locality</label>
                                <select 
                                    value={filters.locality} 
                                    onChange={e => setFilters({...filters, locality: e.target.value})}
                                >
                                    <option value="">All Localities</option>
                                    {localities.map(loc => (
                                        <option key={loc} value={loc}>{loc}</option>
                                    ))}
                                </select>
                            </div>
                            <div className="filter-group">
                                <label>Network Type</label>
                                <select 
                                    value={filters.networkType} 
                                    onChange={e => setFilters({...filters, networkType: e.target.value})}
                                >
                                    <option value="">All Networks</option>
                                    {networkTypes.map(type => (
                                        <option key={type} value={type}>{type}</option>
                                    ))}
                                </select>
                            </div>
                            <div className="filter-group">
                                <label>Min Signal Strength (dBm)</label>
                                <input 
                                    type="number" 
                                    value={filters.minSignalStrength}
                                    onChange={e => setFilters({...filters, minSignalStrength: e.target.value})}
                                    placeholder="e.g., -80"
                                />
                            </div>
                            <div className="filter-group">
                                <label>Max Latency (ms)</label>
                                <input 
                                    type="number" 
                                    value={filters.maxLatency}
                                    onChange={e => setFilters({...filters, maxLatency: e.target.value})}
                                    placeholder="e.g., 100"
                                />
                            </div>
                        </div>
                    )}
                </div>

                {loading && <div className="loading">📊 Processing data and training ML models...</div>}
                
                {error && <div className="error-message">{error}</div>}
                
                {debugInfo && <div className="debug-info">{debugInfo}</div>}
                
                {data.length === 0 && !loading && !error && (
                    <div className="loading">
                        <h3>📡 No Data Loaded</h3>
                        <p>Upload a CSV file or click "Load Sample Data" to get started</p>
                    </div>
                )}

                {filteredData.length > 0 && (
                    <>
                        <div className="stats-grid">
                            <div className="stat-card">
                                <div className="stat-value">{stats.avgSignalStrength}</div>
                                <div className="stat-label">Avg Signal (dBm)</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{stats.avgThroughput}</div>
                                <div className="stat-label">Avg Throughput (Mbps)</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{stats.avgLatency}</div>
                                <div className="stat-label">Avg Latency (ms)</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{stats.avgQuality}%</div>
                                <div className="stat-label">Avg Quality</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{stats.totalPoints}</div>
                                <div className="stat-label">Total Measurements</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{stats.excellentPoints}</div>
                                <div className="stat-label">Excellent Quality</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value">{stats.poorPoints}</div>
                                <div className="stat-label">Poor Quality</div>
                            </div>
                        </div>

                        <div className="dashboard">
                            <div className="card">
                                <h3>📈 Signal Strength Analysis</h3>
                                <div className="chart-container">
                                    <canvas id="signalChart" width="400" height="300"></canvas>
                                    {filteredData.length > 0 && (
                                        <div style={{marginTop: '10px', fontSize: '14px', color: '#666'}}>
                                            Showing {Math.min(15, filteredData.length)} localities
                                        </div>
                                    )}
                                    {filteredData.length === 0 && (
                                        <div style={{textAlign: 'center', padding: '20px', color: '#999'}}>
                                            No data available for chart
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="card">
                                <h3>🎯 Quality vs Throughput</h3>
                                <div className="chart-container">
                                    <canvas id="scatterChart" width="400" height="300"></canvas>
                                    {filteredData.length > 0 && (
                                        <div style={{marginTop: '10px', fontSize: '14px', color: '#666'}}>
                                            Showing {Math.min(50, filteredData.length)} data points
                                        </div>
                                    )}
                                    {filteredData.length === 0 && (
                                        <div style={{textAlign: 'center', padding: '20px', color: '#999'}}>
                                            No data available for chart
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {predictions.length > 0 && (
                            <div className="ml-predictions">
                                <h3>🧠 ML Quality Predictions</h3>
                                <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '15px'}}>
                                    {predictions.map((pred, i) => (
                                        <div key={i} className="prediction-card">
                                            <h4>{pred.locality} ({pred.networkType})</h4>
                                            <p>Actual Quality: {pred.actualQuality.toFixed(1)}%</p>
                                            <p>Predicted Quality: {pred.predictedQuality.toFixed(1)}%</p>
                                            <p>Accuracy: {pred.accuracy.toFixed(1)}%</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        <div className="data-table">
                            <h3>📋 Network Measurements</h3>
                            <div style={{maxHeight: '400px', overflowY: 'auto'}}>
                                <table className="table">
                                    <thead>
                                        <tr>
                                            <th>Locality</th>
                                            <th>Signal (dBm)</th>
                                            <th>Quality (%)</th>
                                            <th>Throughput (Mbps)</th>
                                            <th>Latency (ms)</th>
                                            <th>Network</th>
                                            <th>BB60C</th>
                                            <th>srsRAN</th>
                                            <th>BladeRF</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filteredData.slice(0, 50).map((row, i) => (
                                            <tr key={i}>
                                                <td>{row.locality}</td>
                                                <td>{row.signalStrength.toFixed(1)}</td>
                                                <td>
                                                    <span className={`quality-indicator ${row.signalQuality >= 80 ? 'excellent' : row.signalQuality >= 60 ? 'good' : row.signalQuality >= 40 ? 'fair' : 'poor'}`}>
                                                        {row.signalQuality.toFixed(1)}%
                                                    </span>
                                                </td>
                                                <td>{row.dataThroughput.toFixed(1)}</td>
                                                <td>{row.latency.toFixed(1)}</td>
                                                <td>{row.networkType}</td>
                                                <td>{row.bb60cMeasurement.toFixed(1)}</td>
                                                <td>{row.srsranMeasurement.toFixed(1)}</td>
                                                <td>{row.bladeRFMeasurement.toFixed(1)}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </>
                )}
            </div>
        );
    };

    console.log('Rendering React app...');
    ReactDOM.render(<NetworkAnalyzer />, document.getElementById('root'));
    console.log('React app rendered successfully');

} catch (error) {
    console.error('Application error:', error);
    document.getElementById('root').innerHTML = `
        <div style="text-align: center; padding: 50px; color: white;">
            <h2>Error Loading Application</h2>
            <p>There was an error loading the application. Please check the console for details.</p>
            <p>Error: ${error.message}</p>
            <p>Stack: ${error.stack}</p>
        </div>
    `;
}