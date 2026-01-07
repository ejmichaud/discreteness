#!/usr/bin/env python3
"""
Web app to visualize per-token loss curves across multiple seeds.

Usage:
    python loss_viewer.py logs/aggregated.h5
    python loss_viewer.py logs/aggregated.h5 --port 8080
"""

import argparse
from flask import Flask, render_template_string, jsonify
import h5py
import numpy as np
import tiktoken

app = Flask(__name__)
TOKENIZER = tiktoken.get_encoding("gpt2")
BOS_ID = 50256

# Global state
H5_FILE = None
DOC_NAMES = []
DOC_LENGTHS = []
EVAL_STEPS = []
SEEDS = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Token Loss Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
            background: #fff;
            color: #000;
            padding: 40px 20px;
        }
        
        .page {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .chart-section {
            margin-bottom: 30px;
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .chart-title {
            font-size: 14px;
        }
        
        .chart-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        select {
            font-family: inherit;
            font-size: 14px;
            padding: 8px 12px;
            border: 1px solid #000;
            background: #fff;
            color: #000;
            cursor: pointer;
        }
        
        button {
            font-family: inherit;
            font-size: 12px;
            padding: 6px 10px;
            border: 1px solid #ccc;
            background: #fff;
            color: #000;
            cursor: pointer;
        }
        button:hover { background: #f0f0f0; }
        
        .chart-area {
            height: 375px;
            position: relative;
        }
        
        .chart-area-small {
            height: 220px;
            position: relative;
        }
        
        #chart, #histogram {
            position: absolute;
            top: 0;
            left: 0;
            width: 100% !important;
        }
        
        #chart { height: 375px !important; }
        #histogram { height: 220px !important; }
        
        #chart-placeholder {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #999;
            font-size: 14px;
        }
        
        .stats {
            font-size: 12px;
            color: #666;
            margin-top: 10px;
            text-align: center;
        }
        
        .zoom-hint {
            font-size: 11px;
            color: #999;
            text-align: center;
            margin-top: 5px;
        }
        
        .histogram-section {
            margin-bottom: 30px;
        }
        
        .histogram-title {
            font-size: 13px;
            color: #666;
            margin-bottom: 10px;
        }
        
        .doc-section {
            background: #fff;
            border: 1px solid #000;
            padding: 24px;
        }
        
        .tokens-container {
            line-height: 1.6;
            font-size: 14px;
        }
        
        .token {
            display: inline;
            background: #fff;
            color: #000;
            padding: 2px 0;
            border: 1px solid #ccc;
            border-right: none;
            cursor: pointer;
            transition: background 0.1s;
        }
        .token:last-child, .token.newline { border-right: 1px solid #ccc; }
        .token:hover { background: #f0f0f0; }
        .token.selected { background: #000; color: #fff; border-color: #000; }
        .token.bos { color: #666; font-style: italic; }
        .token.newline { color: #999; }
    </style>
</head>
<body>
    <div class="page">
        <div class="chart-section">
            <div class="chart-header">
                <div class="chart-title" id="chart-title">Click a token to see loss curves ({{ num_seeds }} seeds)</div>
                <div class="chart-controls">
                    <button onclick="resetZoom()" id="reset-btn" style="display:none;">Reset Zoom</button>
                    <select id="doc-select" onchange="loadDocument()">
                        {% for doc in docs %}
                        <option value="{{ loop.index0 }}">{{ doc }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
            <div class="chart-area">
                <canvas id="chart" style="display:none;"></canvas>
                <div id="chart-placeholder"></div>
            </div>
            <div class="zoom-hint" id="zoom-hint" style="display:none;">Drag to zoom · Double-click to reset</div>
            <div class="stats" id="stats"></div>
        </div>
        
        <div class="histogram-section" id="histogram-section" style="display:none;">
            <div class="histogram-title">Final Loss Distribution</div>
            <div class="chart-area-small">
                <canvas id="histogram"></canvas>
            </div>
        </div>
        
        <div class="doc-section">
            <div class="tokens-container" id="tokens"></div>
        </div>
    </div>
    
    <script>
        let chart = null;
        let histogramChart = null;
        let currentDoc = 0;
        let tokensData = [];
        const numSeeds = {{ num_seeds }};
        
        async function loadDocument() {
            currentDoc = parseInt(document.getElementById('doc-select').value);
            const response = await fetch(`/api/tokens/${currentDoc}`);
            tokensData = await response.json();
            renderTokens();
            clearChart();
        }
        
        function renderTokens() {
            const container = document.getElementById('tokens');
            container.innerHTML = '';
            
            tokensData.forEach((tok, idx) => {
                const span = document.createElement('span');
                let classes = 'token';
                if (tok.is_bos) classes += ' bos';
                if (tok.is_newline) classes += ' newline';
                span.className = classes;
                span.textContent = tok.display;
                span.onclick = () => selectToken(idx, span);
                container.appendChild(span);
                
                if (tok.is_newline) {
                    container.appendChild(document.createElement('br'));
                }
            });
        }
        
        function clearChart() {
            document.getElementById('chart').style.display = 'none';
            document.getElementById('chart-placeholder').style.display = 'block';
            document.getElementById('chart-title').textContent = `Click a token to see loss curves (${numSeeds} seeds)`;
            document.getElementById('stats').textContent = '';
            document.getElementById('zoom-hint').style.display = 'none';
            document.getElementById('reset-btn').style.display = 'none';
            document.getElementById('histogram-section').style.display = 'none';
            document.querySelectorAll('.token.selected').forEach(el => el.classList.remove('selected'));
        }
        
        function resetZoom() {
            if (chart) chart.resetZoom();
        }
        
        async function selectToken(idx, element) {
            if (idx === 0) return;
            
            document.querySelectorAll('.token.selected').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            
            const response = await fetch(`/api/loss/${currentDoc}/${idx - 1}`);
            const data = await response.json();
            
            document.getElementById('chart-placeholder').style.display = 'none';
            document.getElementById('chart').style.display = 'block';
            document.getElementById('zoom-hint').style.display = 'block';
            document.getElementById('reset-btn').style.display = 'inline-block';
            document.getElementById('histogram-section').style.display = 'block';
            document.getElementById('chart-title').textContent = 
                `"${tokensData[idx].display}" at position ${idx} (${numSeeds} seeds)`;
            
            // Compute stats across all seeds at final step
            const finalLosses = data.losses.map(l => l[l.length - 1]);
            const mean = finalLosses.reduce((a, b) => a + b, 0) / finalLosses.length;
            const std = Math.sqrt(finalLosses.map(x => (x - mean) ** 2).reduce((a, b) => a + b, 0) / finalLosses.length);
            
            document.getElementById('stats').textContent = 
                `Final step — Mean: ${mean.toFixed(3)} | Std: ${std.toFixed(3)}`;
            
            renderChart(data.steps, data.losses);
            renderHistogram(finalLosses);
        }
        
        function renderChart(steps, allLosses) {
            const ctx = document.getElementById('chart').getContext('2d');
            
            if (chart) chart.destroy();
            
            // Create one dataset per seed, all with low alpha black lines
            const datasets = allLosses.map((losses, i) => ({
                data: losses,
                borderColor: 'rgba(0, 0, 0, 0.08)',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0,
                fill: false
            }));
            
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: steps,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: { 
                        legend: { display: false },
                        zoom: {
                            zoom: {
                                drag: {
                                    enabled: true,
                                    backgroundColor: 'rgba(0,0,0,0.1)',
                                    borderColor: 'rgba(0,0,0,0.3)',
                                    borderWidth: 1
                                },
                                mode: 'xy',
                                onZoomComplete: () => {
                                    document.getElementById('reset-btn').style.display = 'inline-block';
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'logarithmic',
                            title: { display: true, text: 'Step', font: { size: 12 } },
                            grid: { color: '#eee' },
                            ticks: { font: { size: 10 } }
                        },
                        y: {
                            type: 'linear',
                            min: 0,
                            title: { display: true, text: 'Loss (nats)', font: { size: 12 } },
                            grid: { color: '#eee' },
                            ticks: { font: { size: 10 } }
                        }
                    }
                }
            });
        }
        
        function renderHistogram(finalLosses) {
            const ctx = document.getElementById('histogram').getContext('2d');
            
            if (histogramChart) histogramChart.destroy();
            
            // Compute histogram with 30 bins
            const min = Math.min(...finalLosses);
            const max = Math.max(...finalLosses);
            const numBins = 30;
            const binWidth = (max - min) / numBins || 1;
            
            const bins = new Array(numBins).fill(0);
            const binCenters = [];
            
            for (let i = 0; i < numBins; i++) {
                binCenters.push(min + (i + 0.5) * binWidth);
            }
            
            finalLosses.forEach(loss => {
                let binIdx = Math.floor((loss - min) / binWidth);
                if (binIdx >= numBins) binIdx = numBins - 1;
                if (binIdx < 0) binIdx = 0;
                bins[binIdx]++;
            });
            
            histogramChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: binCenters.map(x => x.toFixed(2)),
                    datasets: [{
                        data: bins,
                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                        borderColor: 'rgba(0, 0, 0, 0.9)',
                        borderWidth: 0,
                        barPercentage: 1.0,
                        categoryPercentage: 1.0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            title: { display: true, text: 'Final Loss', font: { size: 11 } },
                            grid: { display: false },
                            ticks: { font: { size: 9 }, maxTicksLimit: 8 }
                        },
                        y: {
                            title: { display: true, text: 'Count', font: { size: 11 } },
                            grid: { color: '#eee' },
                            ticks: { font: { size: 9 } },
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        loadDocument();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, docs=DOC_NAMES, num_seeds=len(SEEDS))


@app.route('/api/tokens/<int:doc_idx>')
def get_tokens(doc_idx):
    """Return tokenized document with display strings."""
    doc_name = DOC_NAMES[doc_idx]
    doc_length = DOC_LENGTHS[doc_idx]
    
    doc_path = f"eval_docs/{doc_name}"
    try:
        with open(doc_path, 'r') as f:
            text = f.read()
        tokens = [BOS_ID] + TOKENIZER.encode(text)
        tokens = tokens[:doc_length + 1]
    except:
        tokens = [BOS_ID] + [0] * doc_length
    
    result = []
    for i, tok_id in enumerate(tokens):
        if tok_id == BOS_ID:
            display = "<BOS>"
            is_bos = True
            is_newline = False
        else:
            decoded = TOKENIZER.decode([tok_id])
            is_bos = False
            is_newline = '\n' in decoded
            if is_newline:
                display = decoded.replace('\n', '↵')
            else:
                display = decoded
        result.append({
            'id': tok_id,
            'display': display,
            'is_bos': is_bos,
            'is_newline': is_newline,
            'position': i
        })
    
    return jsonify(result)


@app.route('/api/loss/<int:doc_idx>/<int:token_idx>')
def get_loss(doc_idx, token_idx):
    """Return loss curves for a specific token across all seeds."""
    doc_name = DOC_NAMES[doc_idx]
    
    # Read from HDF5: [n_seeds, n_steps]
    with h5py.File(H5_FILE, 'r') as f:
        losses = f[f'losses/{doc_name}/token_{token_idx}'][:]
    
    # Filter out step 0 (can't use log(0))
    steps = EVAL_STEPS.copy()
    if steps[0] == 0:
        steps = steps[1:]
        losses = losses[:, 1:]
    
    # Convert to list of lists: [[seed0_losses], [seed1_losses], ...]
    losses_list = losses.astype(float).tolist()
    
    return jsonify({
        'steps': steps,
        'losses': losses_list
    })


def main():
    global H5_FILE, DOC_NAMES, DOC_LENGTHS, EVAL_STEPS, SEEDS
    
    parser = argparse.ArgumentParser(description="Token loss curve viewer (HDF5)")
    parser.add_argument("h5_file", help="Path to aggregated HDF5 file")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    H5_FILE = args.h5_file
    
    print(f"Loading metadata from {H5_FILE}...")
    with h5py.File(H5_FILE, 'r') as f:
        DOC_NAMES = [s.decode() if isinstance(s, bytes) else s for s in f['metadata/doc_names'][:]]
        DOC_LENGTHS = f['metadata/doc_lengths'][:].tolist()
        EVAL_STEPS = f['metadata/eval_steps'][:].tolist()
        SEEDS = f['metadata/seeds'][:].tolist()
    
    print(f"Documents: {DOC_NAMES}")
    print(f"Seeds: {len(SEEDS)}, Steps: {len(EVAL_STEPS)}")
    print(f"\nStarting server at http://{args.host}:{args.port}")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
