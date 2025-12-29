// ==UserScript==
// @name         HuggingFace File Checker
// @namespace    https://github.com/KBerwari/comfyui-huggingface-file-checker
// @version      1.0.0
// @description  Shows which files from HuggingFace repos you have locally
// @author       KBerwari https://github.com/KBerwari/
// @match        https://huggingface.co/*
// @icon         https://huggingface.co/favicon.ico
// @grant        GM_xmlhttpRequest
// @grant        GM_addStyle
// @grant        GM_getValue
// @grant        GM_setValue
// @connect      localhost
// @connect      127.0.0.1
// ==/UserScript==

(function() {
    'use strict';
    
    console.log('HFC: Script starting...');

    // Configuration
    
    const CONFIG = {
        serverUrl: 'http://127.0.0.1:7860',
        checkInterval: 500,  // ms between batch checks
        maxRetries: 3,
        icons: {
            match: '✅',
            mismatch: '⚠️',
            missing: '❌',
            partial: '🟡',  // Some files found, some missing
            unknown: '❓',
            loading: '⏳'
        },
        colors: {
            match: '#22c55e',      // green
            mismatch: '#eab308',   // yellow
            missing: '#ef4444',    // red
            partial: '#f97316',    // orange
            unknown: '#6b7280'     // gray
        }
    };

    // Styles
    
    GM_addStyle(`
        .hfc-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            margin-right: 4px;
            font-size: 12px;
            cursor: help;
            flex-shrink: 0;
            vertical-align: middle;
            line-height: 1;
        }
        
        /* Prevent layout disruption on HF file rows */
        .hfc-icon-wrapper {
            display: inline-flex;
            align-items: center;
            flex-shrink: 0;
            margin-right: 2px;
        }
        
        .hfc-icon-match { color: ${CONFIG.colors.match}; }
        .hfc-icon-mismatch { color: ${CONFIG.colors.mismatch}; }
        .hfc-icon-missing { color: ${CONFIG.colors.missing}; }
        .hfc-icon-partial { color: ${CONFIG.colors.partial}; }
        .hfc-icon-unknown { color: ${CONFIG.colors.unknown}; }
        .hfc-icon-loading { 
            animation: hfc-spin 1s linear infinite;
        }
        
        @keyframes hfc-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .hfc-panel {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 16px;
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #f3f4f6;
            min-width: 280px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }
        
        .hfc-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #374151;
        }
        
        .hfc-panel-title {
            font-weight: 600;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .hfc-panel-close {
            background: none;
            border: none;
            color: #9ca3af;
            cursor: pointer;
            font-size: 18px;
            padding: 0;
            line-height: 1;
        }
        
        .hfc-panel-close:hover {
            color: #f3f4f6;
        }
        
        .hfc-status {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            font-size: 13px;
        }
        
        .hfc-status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .hfc-status-dot.connected { background: ${CONFIG.colors.match}; }
        .hfc-status-dot.disconnected { background: ${CONFIG.colors.missing}; }
        
        .hfc-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-bottom: 12px;
        }
        
        .hfc-stat {
            background: #374151;
            padding: 8px 12px;
            border-radius: 6px;
            text-align: center;
        }
        
        .hfc-stat-value {
            font-size: 18px;
            font-weight: 600;
        }
        
        .hfc-stat-label {
            font-size: 11px;
            color: #9ca3af;
            text-transform: uppercase;
        }
        
        .hfc-btn {
            background: #4f46e5;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            width: 100%;
            margin-top: 8px;
        }
        
        .hfc-btn:hover {
            background: #4338ca;
        }
        
        .hfc-btn:disabled {
            background: #6b7280;
            cursor: not-allowed;
        }
        
        .hfc-btn-secondary {
            background: #374151;
        }
        
        .hfc-btn-secondary:hover {
            background: #4b5563;
        }
        
        .hfc-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #4f46e5;
            color: white;
            border: none;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 20px;
            z-index: 9999;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .hfc-toggle:hover {
            background: #4338ca;
            transform: scale(1.05);
        }
        
        .hfc-tooltip {
            position: absolute;
            background: #1f2937;
            color: #f3f4f6;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: nowrap;
            z-index: 10001;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            pointer-events: none;
        }
        
        .hfc-file-row {
            display: flex;
            align-items: center;
        }
        
        /* Modal styles */
        .hfc-modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 10002;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .hfc-modal {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 12px;
            max-width: 800px;
            max-height: 80vh;
            width: 90%;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .hfc-modal-header {
            padding: 16px 20px;
            border-bottom: 1px solid #374151;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .hfc-modal-title {
            font-size: 16px;
            font-weight: 600;
            color: #f3f4f6;
        }
        
        .hfc-modal-body {
            padding: 16px 20px;
            overflow-y: auto;
            overflow-x: hidden;
            flex: 1;
            min-width: 0;
        }
        
        .hfc-modal-stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }
        
        .hfc-modal-stat {
            background: #374151;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        
        .hfc-modal-stat-value {
            font-size: 24px;
            font-weight: 700;
        }
        
        .hfc-modal-stat-label {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 4px;
        }
        
        .hfc-file-list {
            max-height: 400px;
            overflow-y: auto;
            overflow-x: hidden;
            width: 100%;
        }
        
        .hfc-file-group {
            margin-bottom: 16px;
            max-width: 100%;
            overflow: hidden;
        }
        
        .hfc-file-group-header {
            font-size: 13px;
            font-weight: 600;
            color: #9ca3af;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            word-break: break-word;
            overflow-wrap: break-word;
        }
        
        .hfc-file-group-header:hover {
            color: #f3f4f6;
        }
        
        .hfc-file-item {
            display: flex;
            align-items: center;
            padding: 6px 8px;
            border-radius: 4px;
            font-size: 13px;
            color: #d1d5db;
            max-width: 100%;
            min-width: 0;
        }
        
        .hfc-file-item:hover {
            background: #374151;
        }
        
        .hfc-file-item-icon {
            margin-right: 8px;
            flex-shrink: 0;
        }
        
        .hfc-file-item-name {
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .hfc-file-item-size {
            color: #6b7280;
            font-size: 12px;
            margin-left: 12px;
            flex-shrink: 0;
        }
        
        .hfc-file-item-dir {
            color: #6b7280;
            font-size: 11px;
            margin-left: 8px;
            flex-shrink: 0;
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .hfc-filter-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }
        
        .hfc-filter-tab {
            padding: 6px 12px;
            border-radius: 6px;
            background: #374151;
            color: #9ca3af;
            border: none;
            cursor: pointer;
            font-size: 12px;
        }
        
        .hfc-filter-tab:hover {
            background: #4b5563;
            color: #f3f4f6;
        }
        
        .hfc-filter-tab.active {
            background: #4f46e5;
            color: white;
        }
        
        .hfc-size-info {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #374151;
        }
    `);


    // State
    
    let state = {
        connected: false,
        serverInfo: null,
        fileStatuses: new Map(),  // sha256/filename -> status
        panelVisible: false,
        repoData: null,  // Cached full repo scan data
        lastUrl: null,   // Track URL for navigation detection
        stats: {
            matches: 0,
            mismatches: 0,
            missing: 0,
            total: 0
        }
    };

    // API Functions

    
    function apiRequest(endpoint, method = 'GET', data = null) {
        return new Promise((resolve, reject) => {
            GM_xmlhttpRequest({
                method: method,
                url: `${CONFIG.serverUrl}${endpoint}`,
                headers: {
                    'Content-Type': 'application/json'
                },
                data: data ? JSON.stringify(data) : null,
                onload: function(response) {
                    if (response.status >= 200 && response.status < 300) {
                        try {
                            resolve(JSON.parse(response.responseText));
                        } catch (e) {
                            resolve(response.responseText);
                        }
                    } else {
                        reject(new Error(`HTTP ${response.status}`));
                    }
                },
                onerror: function(error) {
                    reject(error);
                },
                ontimeout: function() {
                    reject(new Error('Request timeout'));
                }
            });
        });
    }

    async function checkServerStatus() {
        try {
            const status = await apiRequest('/api/status');
            state.connected = true;
            state.serverInfo = status;
            return true;
        } catch (e) {
            state.connected = false;
            state.serverInfo = null;
            return false;
        }
    }

    async function checkFiles(files) {
        if (!state.connected) return null;
        
        try {
            const response = await apiRequest('/api/check/batch', 'POST', { files });
            return response.results;
        } catch (e) {
            console.error('HFC: Error checking files:', e);
            return null;
        }
    }


    // File Detection
    
    function findAllFileRows() {
        const rows = [];
        
        // Method 1: Links to /blob/ (file view links)
        document.querySelectorAll('a[href*="/blob/"]').forEach(link => {
            const href = link.getAttribute('href') || '';
            
            // Skip if it's a directory or already processed
            if (href.includes('/tree/')) return;
            
            // Extract filename from URL
            const parts = href.split('/blob/');
            if (parts.length < 2) return;
            
            const pathPart = parts[1].split('/').slice(1).join('/'); // Remove branch name
            // Decode the full path first, then extract filename
            const decodedPath = decodeURIComponent(pathPart);
            const filename = decodedPath.split('/').pop();
            
            if (!filename || filename.includes('..')) return;
            
            console.log('HFC: File detected:', { href, pathPart, decodedPath, filename });
            
            // Find the row container
            const row = link.closest('li, tr, .group, [class*="row"], [class*="Row"]') || link.parentElement;
            if (!row || row.dataset.hfcProcessed) return;
            
            rows.push({
                filename: filename,
                fullPath: decodedPath,
                sha256: null,
                element: row,
                link: link
            });
            row.dataset.hfcProcessed = 'true';
        });
        
        // Method 2: Look for file size indicators
        if (rows.length === 0) {
            document.querySelectorAll('a[class*="Link"]').forEach(link => {
                const href = link.getAttribute('href') || '';
                if (!href.includes('/blob/') && !href.includes('/resolve/')) return;
                
                const filename = decodeURIComponent(href.split('/').pop());
                if (!filename) return;
                
                const row = link.closest('li, tr, div[class*="group"]') || link.parentElement;
                if (!row || row.dataset.hfcProcessed) return;
                
                rows.push({
                    filename: filename,
                    sha256: null,
                    element: row,
                    link: link
                });
                row.dataset.hfcProcessed = 'true';
            });
        }
        
        console.log('HFC: Found', rows.length, 'file rows');
        return rows;
    }

    function findAllFolderRows() {
        const folders = [];
        
        document.querySelectorAll('a[href*="/tree/"]').forEach(link => {
            const href = link.getAttribute('href') || '';
            
            if (!href.includes('/tree/')) return;
            
            // URL format: /user/repo/tree/branch/folder/path
            const parts = href.split('/tree/');
            if (parts.length < 2) return;
            
            const afterTree = parts[1].split('/');
            if (afterTree.length < 2) return; // Just branch, no folder
            
            const folderPath = afterTree.slice(1).join('/'); // Remove branch name
            if (!folderPath) return;
            
            const folderName = decodeURIComponent(afterTree[afterTree.length - 1]);
            
            // Find the row container
            const row = link.closest('li, tr, .group, [class*="row"], [class*="Row"]') || link.parentElement;
            if (!row || row.dataset.hfcFolderProcessed) return;
            
            folders.push({
                folderName: folderName,
                folderPath: folderPath,
                element: row,
                link: link
            });
            row.dataset.hfcFolderProcessed = 'true';
        });
        
        console.log('HFC: Found', folders.length, 'folder rows');
        return folders;
    }

    function calculateFolderStatus(folderPath, allFiles) {
        const filesInFolder = allFiles.filter(f => {
            const filePath = f.path || f.filename;
            return filePath.startsWith(folderPath + '/') || filePath === folderPath;
        });
        
        if (filesInFolder.length === 0) {
            return { status: 'unknown', found: 0, missing: 0, mismatch: 0, total: 0 };
        }
        
        const found = filesInFolder.filter(f => f.status === 'found').length;
        const missing = filesInFolder.filter(f => f.status === 'missing').length;
        const mismatch = filesInFolder.filter(f => f.status === 'mismatch').length;
        const total = filesInFolder.length;
        
        let status;
        if (found === total) {
            status = 'match';  // All files found
        } else if (missing === total) {
            status = 'missing';  // All files missing
        } else if (found > 0 || mismatch > 0) {
            status = 'partial';  // Some found, some missing
        } else {
            status = 'missing';
        }
        
        return { status, found, missing, mismatch, total };
    }

    // UI Functions
    
    function createIcon(status, title = '') {
        const span = document.createElement('span');
        span.className = `hfc-icon hfc-icon-${status}`;
        span.textContent = CONFIG.icons[status] || CONFIG.icons.unknown;
        span.title = title;
        return span;
    }

    function updateFileIcon(row, status, title, link) {
        row.querySelectorAll('.hfc-icon').forEach(icon => icon.remove());
        
        if (link && link.parentElement) {
            link.parentElement.querySelectorAll('.hfc-icon').forEach(icon => icon.remove());
        }
        
        const icon = createIcon(status, title);
        
        if (link) {
            link.insertBefore(icon, link.firstChild);
        } else {
            const container = row.querySelector('td:first-child, .flex, a, span');
            if (container) {
                if (container.tagName === 'A') {
                    container.insertBefore(icon, container.firstChild);
                } else {
                    container.insertBefore(icon, container.firstChild);
                }
            } else {
                row.insertBefore(icon, row.firstChild);
            }
        }
    }

    function createPanel() {
        const panel = document.createElement('div');
        panel.className = 'hfc-panel';
        panel.id = 'hfc-panel';
        panel.innerHTML = `
            <div class="hfc-panel-header">
                <div class="hfc-panel-title">
                    📁 HF File Checker
                </div>
                <button class="hfc-panel-close" id="hfc-close">×</button>
            </div>
            
            <div class="hfc-status">
                <span class="hfc-status-dot ${state.connected ? 'connected' : 'disconnected'}"></span>
                <span id="hfc-status-text">${state.connected ? 'Connected to local server' : 'Server not running'}</span>
            </div>
            
            <div class="hfc-stats">
                <div class="hfc-stat">
                    <div class="hfc-stat-value" style="color: ${CONFIG.colors.match}" id="hfc-stat-match">0</div>
                    <div class="hfc-stat-label">Have</div>
                </div>
                <div class="hfc-stat">
                    <div class="hfc-stat-value" style="color: ${CONFIG.colors.missing}" id="hfc-stat-missing">0</div>
                    <div class="hfc-stat-label">Missing</div>
                </div>
                <div class="hfc-stat">
                    <div class="hfc-stat-value" style="color: ${CONFIG.colors.mismatch}" id="hfc-stat-mismatch">0</div>
                    <div class="hfc-stat-label">Different</div>
                </div>
                <div class="hfc-stat">
                    <div class="hfc-stat-value" id="hfc-stat-total">0</div>
                    <div class="hfc-stat-label">Total</div>
                </div>
            </div>
            
            <button class="hfc-btn" id="hfc-full-repo">🔍 Scan Full Repo</button>
            <button class="hfc-btn hfc-btn-secondary" id="hfc-rescan">🔄 Rescan This Page</button>
            <button class="hfc-btn hfc-btn-secondary" id="hfc-server-rescan">📂 Rescan Local Files</button>
        `;
        
        document.body.appendChild(panel);
        
        // Event listeners
        document.getElementById('hfc-close').onclick = () => togglePanel(false);
        document.getElementById('hfc-rescan').onclick = () => scanCurrentPage();
        document.getElementById('hfc-server-rescan').onclick = rescanServer;
        document.getElementById('hfc-full-repo').onclick = scanFullRepo;
        
        return panel;
    }

    function createToggleButton() {
        const btn = document.createElement('button');
        btn.className = 'hfc-toggle';
        btn.id = 'hfc-toggle';
        btn.textContent = '📁';
        btn.title = 'HuggingFace File Checker';
        btn.onclick = () => togglePanel(!state.panelVisible);
        document.body.appendChild(btn);
        return btn;
    }

    function togglePanel(show) {
        state.panelVisible = show;
        
        let panel = document.getElementById('hfc-panel');
        let toggle = document.getElementById('hfc-toggle');
        
        if (show) {
            if (!panel) panel = createPanel();
            panel.style.display = 'block';
            if (toggle) toggle.style.display = 'none';
            updatePanelStats();
        } else {
            if (panel) panel.style.display = 'none';
            if (!toggle) toggle = createToggleButton();
            toggle.style.display = 'flex';
        }
    }

    function updatePanelStats() {
        const el = (id) => document.getElementById(id);
        
        if (el('hfc-stat-match')) el('hfc-stat-match').textContent = state.stats.matches;
        if (el('hfc-stat-missing')) el('hfc-stat-missing').textContent = state.stats.missing;
        if (el('hfc-stat-mismatch')) el('hfc-stat-mismatch').textContent = state.stats.mismatches;
        if (el('hfc-stat-total')) el('hfc-stat-total').textContent = state.stats.total;
        
        const statusDot = document.querySelector('.hfc-status-dot');
        const statusText = el('hfc-status-text');
        
        if (statusDot) {
            statusDot.className = `hfc-status-dot ${state.connected ? 'connected' : 'disconnected'}`;
        }
        if (statusText) {
            if (state.connected && state.serverInfo) {
                statusText.textContent = `Connected (${state.serverInfo.cache?.total_files || 0} local files)`;
            } else {
                statusText.textContent = 'Server not running - start with: python main.py server';
            }
        }
    }


    // Main Logic

    let isScanning = false;  // Prevent concurrent scans
    
    async function scanCurrentPage() {
        if (isScanning) {
            console.log('HFC: Scan already in progress, skipping');
            return;
        }
        isScanning = true;
        
        try {
            // Reset stats
            state.stats = { matches: 0, mismatches: 0, missing: 0, total: 0 };
            
            // Clear ALL existing icons first
            document.querySelectorAll('.hfc-icon').forEach(icon => icon.remove());
            
            // Clear processed flags
            document.querySelectorAll('[data-hfc-processed]').forEach(el => {
                delete el.dataset.hfcProcessed;
            });
            document.querySelectorAll('[data-hfc-folder-processed]').forEach(el => {
                delete el.dataset.hfcFolderProcessed;
            });
            
            // Check server status
            await checkServerStatus();
            updatePanelStats();
            
            if (!state.connected) {
                console.log('HFC: Server not connected');
                return;
            }
            
            // Find all file rows and folder rows
            const rows = findAllFileRows();
            const folders = findAllFolderRows();
            
            console.log('HFC: Found', rows.length, 'files and', folders.length, 'folders');
            
            // Add loading icons to files
            rows.forEach(row => {
                updateFileIcon(row.element, 'loading', 'Checking...', row.link);
            });
            
            // Add loading icons to folders
            folders.forEach(folder => {
                updateFileIcon(folder.element, 'loading', 'Checking folder contents...', folder.link);
            });
            
            // If there are folders, we need full repo data
            if (folders.length > 0 && !state.repoData) {
                const repoInfo = getRepoInfoFromUrl();
                if (repoInfo) {
                    console.log('HFC: Fetching full repo data for folder status...');
                    try {
                        state.repoData = await apiRequest('/api/check/repo', 'POST', {
                            repo_id: repoInfo.repoId,
                            repo_type: repoInfo.repoType,
                            revision: repoInfo.revision
                        });
                        console.log('HFC: Got full repo data:', state.repoData?.total_files, 'files');
                    } catch (e) {
                        console.error('HFC: Error fetching full repo data:', e);
                    }
                }
            }
            
            // Update folder icons using cached repo data
            if (state.repoData && state.repoData.files) {
                folders.forEach(folder => {
                    const folderStats = calculateFolderStatus(folder.folderPath, state.repoData.files);
                    const tooltip = `${folderStats.found}/${folderStats.total} files found locally`;
                    updateFileIcon(folder.element, folderStats.status, tooltip, folder.link);
                });
            } else {
                // No repo data, mark folders as unknown
                folders.forEach(folder => {
                    updateFileIcon(folder.element, 'unknown', 'Click "Scan Full Repo" to check folder contents', folder.link);
                });
            }
            
            // Check individual files on this page
            if (rows.length > 0) {
                state.stats.total = rows.length;
                
                // Prepare batch request - try to get SHA256 from cached repo data if available
                const files = rows.map(r => {
                    let sha256 = r.sha256;
                    
                    // If we have cached repo data, look up the SHA256 by path
                    if (!sha256 && state.repoData && state.repoData.files) {
                        const repoFile = state.repoData.files.find(f => 
                            f.path === r.fullPath || 
                            f.path === r.filename ||
                            f.filename === r.filename ||
                            (f.path && f.path.endsWith('/' + r.filename))
                        );
                        if (repoFile && repoFile.sha256) {
                            sha256 = repoFile.sha256;
                            r.sha256 = sha256;
                            console.log('HFC: Found SHA256 from cache for', r.filename, ':', sha256.substring(0, 16) + '...');
                        }
                    }
                    
                    return {
                        filename: r.filename,
                        fullPath: r.fullPath,
                        sha256: sha256
                    };
                });
                
                console.log('HFC: Checking files:', files.slice(0, 5));
                
                const results = await checkFiles(files);
                
                console.log('HFC: Results:', results);
                
                if (!results) {
                    rows.forEach(row => {
                        updateFileIcon(row.element, 'unknown', 'Could not check', row.link);
                    });
                } else {
                    // Update icons based on results
                    rows.forEach(row => {
                        const filename = (row.filename || '').trim().toLowerCase();
                        const sha256 = (row.sha256 || '').toLowerCase();
                        
                        let result = null;
                        
                        if (sha256) {
                            result = results[sha256];
                        }
                        
                        if (!result) {
                            result = results[filename];
                        }
                        
                        console.log('HFC: Looking up', { filename, sha256: sha256?.substring(0,16), found: result?.found });
                        
                        if (result && result.found) {
                            if (result.mismatch) {
                                updateFileIcon(row.element, 'mismatch', `Different version locally in: ${result.directories?.join(', ')}`, row.link);
                                state.stats.mismatches++;
                            } else {
                                updateFileIcon(row.element, 'match', `Found in: ${result.directories?.join(', ')}`, row.link);
                                state.stats.matches++;
                            }
                        } else {
                            updateFileIcon(row.element, 'missing', 'Not found locally', row.link);
                            state.stats.missing++;
                        }
                    });
                }
            }
            
            updatePanelStats();
            console.log('HFC: Scan complete', state.stats);
        } finally {
            isScanning = false;
        }
    }

    async function rescanServer() {
        if (!state.connected) return;
        
        const btn = document.getElementById('hfc-server-rescan');
        if (btn) {
            btn.disabled = true;
            btn.textContent = '⏳ Rescanning...';
        }
        
        try {
            await apiRequest('/api/rescan', 'POST');
            await checkServerStatus();
            
            // Clear cached repo data so it gets refreshed
            state.repoData = null;
            
            updatePanelStats();
            
            // Re-scan the page with new data
            await scanCurrentPage();
        } catch (e) {
            console.error('HFC: Error rescanning server:', e);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = '📂 Rescan Local Files';
            }
        }
    }

    async function autoFetchRepoData() {
        // Automatically fetch full repo data for SHA256 matching
        if (!state.connected) return;
        
        const repoInfo = getRepoInfoFromUrl();
        if (!repoInfo) {
            console.log('HFC: Could not detect repo from URL, skipping auto-fetch');
            return;
        }
        
        // Skip if we already have data for this repo
        if (state.repoData && state.repoData.repo_id === repoInfo.repoId) {
            console.log('HFC: Already have repo data for', repoInfo.repoId);
            return;
        }
        
        console.log('HFC: Auto-fetching repo data for', repoInfo.repoId);
        
        try {
            const result = await apiRequest('/api/check/repo', 'POST', {
                repo_id: repoInfo.repoId,
                repo_type: repoInfo.repoType,
                revision: repoInfo.revision
            });
            
            state.repoData = result;
            console.log('HFC: Auto-fetched', result.total_files, 'files from repo');
            
            // Update stats from full repo data
            state.stats = {
                matches: result.found || 0,
                mismatches: result.mismatch || 0,
                missing: result.missing || 0,
                total: result.total_files || 0
            };
            updatePanelStats();
            
        } catch (e) {
            console.error('HFC: Error auto-fetching repo data:', e);
        }
    }

    async function updateFolderIcons() {
        // Clear folder processed flags
        document.querySelectorAll('[data-hfc-folder-processed]').forEach(el => {
            delete el.dataset.hfcFolderProcessed;
        });
        
        const folders = findAllFolderRows();
        
        if (state.repoData && state.repoData.files) {
            folders.forEach(folder => {
                const folderStats = calculateFolderStatus(folder.folderPath, state.repoData.files);
                const tooltip = `${folderStats.found}/${folderStats.total} files found locally`;
                updateFileIcon(folder.element, folderStats.status, tooltip, folder.link);
            });
        }
    }

    function formatSize(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return (bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function getRepoInfoFromUrl() {
        const path = window.location.pathname;
        const parts = path.split('/').filter(p => p);
        
        let repoType = 'model';
        let startIdx = 0;
        
        if (parts[0] === 'datasets') {
            repoType = 'dataset';
            startIdx = 1;
        } else if (parts[0] === 'spaces') {
            repoType = 'space';
            startIdx = 1;
        }
        
        if (parts.length < startIdx + 2) return null;
        
        const owner = parts[startIdx];
        const repo = parts[startIdx + 1];
        
        let revision = 'main';
        const treeIdx = parts.indexOf('tree');
        if (treeIdx !== -1 && parts[treeIdx + 1]) {
            revision = parts[treeIdx + 1];
        }
        
        return {
            repoId: `${owner}/${repo}`,
            repoType,
            revision,
            url: window.location.href
        };
    }

    async function scanFullRepo() {
        if (!state.connected) {
            alert('Server not connected. Start with: python main.py server');
            return;
        }
        
        const repoInfo = getRepoInfoFromUrl();
        if (!repoInfo) {
            alert('Could not detect repository from URL');
            return;
        }
        
        const btn = document.getElementById('hfc-full-repo');
        if (btn) {
            btn.disabled = true;
            btn.textContent = '⏳ Scanning...';
        }
        
        try {
            console.log('HFC: Scanning full repo:', repoInfo);
            const result = await apiRequest('/api/check/repo', 'POST', {
                repo_id: repoInfo.repoId,
                repo_type: repoInfo.repoType,
                revision: repoInfo.revision
            });
            
            console.log('HFC: Full repo scan result:', result);
            
            // Cache the result for folder status
            state.repoData = result;
            
            // Refresh folder icons with the new data
            await updateFolderIcons();
            
            showRepoModal(result);
            
        } catch (e) {
            console.error('HFC: Error scanning full repo:', e);
            alert('Error scanning repository: ' + e.message);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = '🔍 Scan Full Repo';
            }
        }
    }

    function showRepoModal(result) {
        const existing = document.getElementById('hfc-modal-overlay');
        if (existing) existing.remove();
        
        const overlay = document.createElement('div');
        overlay.className = 'hfc-modal-overlay';
        overlay.id = 'hfc-modal-overlay';
        
        const foundFiles = result.files.filter(f => f.status === 'found');
        const missingFiles = result.files.filter(f => f.status === 'missing');
        const mismatchFiles = result.files.filter(f => f.status === 'mismatch');
        
        overlay.innerHTML = `
            <div class="hfc-modal">
                <div class="hfc-modal-header">
                    <div class="hfc-modal-title">📦 ${escapeHtml(result.repo_id)}</div>
                    <button class="hfc-panel-close" id="hfc-modal-close">×</button>
                </div>
                <div class="hfc-modal-body">
                    <div class="hfc-modal-stats">
                        <div class="hfc-modal-stat">
                            <div class="hfc-modal-stat-value">${result.total_files}</div>
                            <div class="hfc-modal-stat-label">Total Files</div>
                        </div>
                        <div class="hfc-modal-stat">
                            <div class="hfc-modal-stat-value" style="color: ${CONFIG.colors.match}">${result.found}</div>
                            <div class="hfc-modal-stat-label">Have Locally</div>
                        </div>
                        <div class="hfc-modal-stat">
                            <div class="hfc-modal-stat-value" style="color: ${CONFIG.colors.missing}">${result.missing}</div>
                            <div class="hfc-modal-stat-label">Missing</div>
                        </div>
                        <div class="hfc-modal-stat">
                            <div class="hfc-modal-stat-value" style="color: ${CONFIG.colors.mismatch}">${result.mismatch}</div>
                            <div class="hfc-modal-stat-label">Different</div>
                        </div>
                    </div>
                    
                    <div class="hfc-filter-tabs">
                        <button class="hfc-filter-tab active" data-filter="all">All (${result.total_files})</button>
                        <button class="hfc-filter-tab" data-filter="found">✅ Have (${result.found})</button>
                        <button class="hfc-filter-tab" data-filter="missing">❌ Missing (${result.missing})</button>
                        <button class="hfc-filter-tab" data-filter="mismatch">⚠️ Different (${result.mismatch})</button>
                    </div>
                    
                    <div class="hfc-file-list" id="hfc-file-list">
                        ${renderFileList(result.files, 'all')}
                    </div>
                    
                    <div class="hfc-size-info">
                        <strong>Have locally:</strong> ${formatSize(result.found_size)} &nbsp;|&nbsp;
                        <strong>Need to download:</strong> ${formatSize(result.missing_size)}
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(overlay);
        
        document.getElementById('hfc-modal-close').onclick = () => overlay.remove();
        overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
        
        overlay.querySelectorAll('.hfc-filter-tab').forEach(tab => {
            tab.onclick = () => {
                overlay.querySelectorAll('.hfc-filter-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const filter = tab.dataset.filter;
                document.getElementById('hfc-file-list').innerHTML = renderFileList(result.files, filter);
            };
        });
        
        overlay.dataset.files = JSON.stringify(result.files);
    }

    function renderFileList(files, filter) {
        let filtered = files;
        if (filter !== 'all') {
            filtered = files.filter(f => f.status === filter);
        }
        
        if (filtered.length === 0) {
            return '<div style="color: #6b7280; text-align: center; padding: 20px;">No files in this category</div>';
        }
        
        // Group by folder
        const byFolder = {};
        filtered.forEach(file => {
            const pathParts = (file.path || file.filename).split('/');
            const folder = pathParts.length > 1 ? pathParts.slice(0, -1).join('/') : '/';
            if (!byFolder[folder]) byFolder[folder] = [];
            byFolder[folder].push(file);
        });
        
        let html = '';
        const folders = Object.keys(byFolder).sort();
        
        for (const folder of folders) {
            const folderFiles = byFolder[folder];
            const folderLabel = escapeHtml(folder === '/' ? 'Root' : folder);
            
            let filesHtml = '';
            for (const file of folderFiles) {
                const icon = getStatusIcon(file.status);
                const name = escapeHtml(file.filename);
                const path = escapeHtml(file.path || file.filename);
                const size = formatSize(file.size);
                const dirsTitle = file.directories?.length ? escapeHtml(file.directories.join(', ')) : '';
                const dirsFirst = file.directories?.length ? escapeHtml(file.directories[0]) : '';
                const dirs = file.directories?.length ? `<span class="hfc-file-item-dir" title="${dirsTitle}">in ${dirsFirst}${file.directories.length > 1 ? '...' : ''}</span>` : '';
                filesHtml += `<div class="hfc-file-item"><span class="hfc-file-item-icon">${icon}</span><span class="hfc-file-item-name" title="${path}">${name}</span><span class="hfc-file-item-size">${size}</span>${dirs}</div>`;
            }
            
            html += `<div class="hfc-file-group"><div class="hfc-file-group-header">📁 ${folderLabel} (${folderFiles.length})</div>${filesHtml}</div>`;
        }
        
        return html;
    }

    function getStatusIcon(status) {
        switch(status) {
            case 'found': return CONFIG.icons.match;
            case 'missing': return CONFIG.icons.missing;
            case 'mismatch': return CONFIG.icons.mismatch;
            default: return CONFIG.icons.unknown;
        }
    }

    // Initialization

    
    async function init() {
        console.log('HFC: Initializing HuggingFace File Checker');
        console.log('HFC: Current path:', window.location.pathname);
        
        const pathParts = window.location.pathname.split('/').filter(p => p);
        const isRepoPage = pathParts.length >= 2;  // At least /user/repo
        const isFileBrowser = window.location.pathname.includes('/tree/') || isRepoPage;
        
        console.log('HFC: Is repo page:', isRepoPage, 'Is file browser:', isFileBrowser);
        
        if (!isFileBrowser) {
            console.log('HFC: Not a file browser page, skipping');
            return;
        }
        
        // Create toggle button
        createToggleButton();
        
        // Check server and scan page
        await checkServerStatus();
        console.log('HFC: Server connected:', state.connected);
        
        if (state.connected) {
            // Auto-fetch full repo data first (for SHA256 matching), then scan page
            setTimeout(async () => {
                await autoFetchRepoData();
                await scanCurrentPage();
            }, 1500);
        } else {
            console.log('HFC: Server not running. Start with: python main.py server');
        }
        
        const observer = new MutationObserver((mutations) => {
            clearTimeout(window.hfcDebounce);
            window.hfcDebounce = setTimeout(async () => {
                if (state.connected && !isScanning) {
                    const currentUrl = window.location.href;
                    if (currentUrl !== state.lastUrl) {
                        state.lastUrl = currentUrl;
                        const repoInfo = getRepoInfoFromUrl();
                        if (repoInfo && state.repoData && state.repoData.repo_id !== repoInfo.repoId) {
                            console.log('HFC: Repo changed, fetching new repo data');
                            state.repoData = null;
                            await autoFetchRepoData();
                        }
                    }
                    
                    const rows = findAllFileRows();
                    const unprocessed = rows.filter(r => !r.element.querySelector('.hfc-icon'));
                    if (unprocessed.length > 0) {
                        console.log('HFC: Found', unprocessed.length, 'new rows, rescanning');
                        scanCurrentPage();
                    }
                }
            }, 500);
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    console.log('HFC: Setting up init, readyState:', document.readyState);
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();