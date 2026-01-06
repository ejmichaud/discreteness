#!/usr/bin/env python3
"""
Web app to visualize per-token loss curves from eval tracker output.

Usage:
    python loss_viewer.py logs/run_id_eval_losses.pt
    python loss_viewer.py logs/run_id_eval_losses.pt --port 8080
"""

import argparse
import json
from flask import Flask, render_template_string, jsonify
import torch
import tiktoken

app = Flask(__name__)
TOKENIZER = tiktoken.get_encoding("gpt2")
BOS_ID = 50256

# Global state loaded from file
DATA = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Token Loss Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
        
        /* Chart section - top */
        .chart-section {
            margin-bottom: 40px;
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
        
        select {
            font-family: inherit;
            font-size: 14px;
            padding: 8px 12px;
            border: 1px solid #000;
            background: #fff;
            color: #000;
            cursor: pointer;
        }
        
        .chart-area {
            height: 300px;
            position: relative;
        }
        
        #chart {
            position: absolute;
            top: 0;
            left: 0;
            width: 100% !important;
            height: 300px !important;
        }
        
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
        
        /* Document section - bottom */
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
        <!-- Chart at top -->
        <div class="chart-section">
            <div class="chart-header">
                <div class="chart-title" id="chart-title">Click a token to see its loss curve</div>
                <select id="doc-select" onchange="loadDocument()">
                    {% for doc in docs %}
                    <option value="{{ loop.index0 }}">{{ doc }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="chart-area">
                <canvas id="chart" style="display:none; width:100%; height:100%;"></canvas>
                <div id="chart-placeholder"></div>
            </div>
            <div class="stats" id="stats"></div>
        </div>
        
        <!-- Document below -->
        <div class="doc-section">
            <div class="tokens-container" id="tokens"></div>
        </div>
    </div>
    
    <script>
        let chart = null;
        let currentDoc = 0;
        let tokensData = [];
        
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
                
                // Add actual line break after newline tokens
                if (tok.is_newline) {
                    container.appendChild(document.createElement('br'));
                }
            });
        }
        
        function clearChart() {
            document.getElementById('chart').style.display = 'none';
            document.getElementById('chart-placeholder').style.display = 'block';
            document.getElementById('chart-title').textContent = 'Click a token to see its loss curve';
            document.getElementById('stats').textContent = '';
            document.querySelectorAll('.token.selected').forEach(el => el.classList.remove('selected'));
        }
        
        async function selectToken(idx, element) {
            // Can't show loss for first token (no prediction)
            if (idx === 0) return;
            
            document.querySelectorAll('.token.selected').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            
            const response = await fetch(`/api/loss/${currentDoc}/${idx - 1}`);
            const data = await response.json();
            
            document.getElementById('chart-placeholder').style.display = 'none';
            document.getElementById('chart').style.display = 'block';
            document.getElementById('chart-title').textContent = 
                `"${tokensData[idx].display}" at position ${idx}`;
            
            const stats = `Final: ${data.losses[data.losses.length-1].toFixed(3)} | ` +
                         `Min: ${Math.min(...data.losses).toFixed(3)} | ` +
                         `Max: ${Math.max(...data.losses).toFixed(3)}`;
            document.getElementById('stats').textContent = stats;
            
            renderChart(data.steps, data.losses);
        }
        
        function renderChart(steps, losses) {
            const ctx = document.getElementById('chart').getContext('2d');
            
            if (chart) chart.destroy();
            
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: steps,
                    datasets: [{
                        data: losses,
                        borderColor: '#000',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
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
        
        // Load first document on page load
        loadDocument();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, docs=DATA['doc_names'])


@app.route('/api/tokens/<int:doc_idx>')
def get_tokens(doc_idx):
    """Return tokenized document with display strings."""
    # Reconstruct tokens from the stored data
    doc_name = DATA['doc_names'][doc_idx]
    doc_length = DATA['doc_lengths'][doc_idx]
    
    # We need to get the actual tokens - reconstruct from first step's data
    # The losses are for positions 1..N (predicting token i from tokens 0..i-1)
    # So we have doc_length loss values, meaning doc_length+1 tokens total
    
    # Load the original document to get tokens
    doc_path = f"eval_docs/{doc_name}"
    try:
        with open(doc_path, 'r') as f:
            text = f.read()
        tokens = [BOS_ID] + TOKENIZER.encode(text)
        tokens = tokens[:doc_length + 1]  # Match stored length
    except:
        # Fallback: create placeholder tokens
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
            # Check if token contains newline
            is_newline = '\n' in decoded
            if is_newline:
                # Show carriage return symbol for newlines
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
    """Return loss curve for a specific token position."""
    steps = sorted(DATA['results'].keys())
    losses = []
    
    for step in steps:
        doc_losses = DATA['results'][step][doc_idx]
        if token_idx < len(doc_losses):
            losses.append(float(doc_losses[token_idx]))
        else:
            losses.append(None)
    
    # Filter out steps with None and step 0 (can't use log(0))
    valid = [(s, l) for s, l in zip(steps, losses) if l is not None and s > 0]
    steps, losses = zip(*valid) if valid else ([], [])
    
    return jsonify({
        'steps': list(steps),
        'losses': list(losses)
    })


def main():
    global DATA
    
    parser = argparse.ArgumentParser(description="Token loss curve viewer")
    parser.add_argument("losses_file", help="Path to eval losses .pt file")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    print(f"Loading {args.losses_file}...")
    DATA = torch.load(args.losses_file, map_location='cpu')
    
    # Convert tensor keys to int if needed
    if DATA['results']:
        first_key = next(iter(DATA['results']))
        if isinstance(first_key, torch.Tensor):
            DATA['results'] = {int(k): v for k, v in DATA['results'].items()}
    
    print(f"Loaded {len(DATA['doc_names'])} documents, {len(DATA['results'])} steps")
    print(f"Documents: {DATA['doc_names']}")
    print(f"\nStarting server at http://{args.host}:{args.port}")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

