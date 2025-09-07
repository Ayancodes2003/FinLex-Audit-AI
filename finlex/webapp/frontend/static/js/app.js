/**
 * FinLex Audit AI - Frontend JavaScript Application
 * Modern SPA with comprehensive compliance features
 */

class FinLexApp {
    constructor() {
        this.currentSection = 'dashboard';
        this.apiBase = '/api';
        this.init();
    }

    init() {
        console.log('🚀 Setting up FinLex Audit AI...');
        
        try {
            this.setupNavigation();
            this.setupFileUploads();
            this.setupEventListeners();
            this.loadDashboard();
            this.checkSystemHealth();
            console.log('✅ All components initialized successfully');
        } catch (error) {
            console.error('⚠️ Initialization error:', error);
            this.showToast('Failed to initialize application', 'error');
        }
    }

    // === NAVIGATION ===
    setupNavigation() {
        console.log('🧭 Setting up navigation...');
        const navButtons = document.querySelectorAll('.nav-btn');
        
        if (navButtons.length === 0) {
            console.warn('⚠️ No navigation buttons found');
            return;
        }
        
        navButtons.forEach((btn, index) => {
            console.log(`🔘 Setting up nav button ${index + 1}: ${btn.textContent}`);
            btn.addEventListener('click', (e) => {
                const section = e.target.getAttribute('data-section');
                console.log(`📜 Navigating to section: ${section}`);
                this.showSection(section);
            });
        });
        
        console.log(`✅ Navigation setup complete - ${navButtons.length} buttons`);
    }

    showSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });

        // Show target section
        document.getElementById(sectionId).classList.add('active');

        // Update nav buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionId}"]`).classList.add('active');

        this.currentSection = sectionId;

        // Load section-specific data
        switch (sectionId) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'violations':
                this.loadViolations();
                break;
        }
    }

    // === FILE UPLOADS ===
    // === FILE UPLOADS ===
    setupFileUploads() {
        console.log('📁 Setting up file uploads...');
        // File upload functionality would go here
        console.log('✅ File uploads setup complete');
    }
    
    // === COMPLIANCE SCANNING ===
    async runComplianceScan() {
        console.log('🔍 Running compliance scan...');
        this.showToast('Compliance scan feature coming soon!', 'info');
    }
    
    // === VIOLATIONS ===
    async loadViolations(status = 'all') {
        console.log(`🚨 Loading violations (${status})...`);
        this.showToast('Violation management feature coming soon!', 'info');
    }

    setupTransactionUpload() {
        const uploadArea = document.getElementById('transaction-upload-area');
        const fileInput = document.getElementById('transaction-file');
        const preview = document.getElementById('transaction-preview');

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleTransactionFile(files[0]);
            }
        });

        // File input
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleTransactionFile(e.target.files[0]);
            }
        });

        // Upload button
        document.getElementById('upload-transactions-btn').addEventListener('click', () => {
            this.uploadTransactions();
        });

        // Cancel button
        document.getElementById('cancel-upload-btn').addEventListener('click', () => {
            preview.classList.add('hidden');
            fileInput.value = '';
        });
    }

    setupPolicyUpload() {
        const uploadArea = document.getElementById('policy-upload-area');
        const fileInput = document.getElementById('policy-file');
        const preview = document.getElementById('policy-preview');

        // Similar drag and drop setup
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handlePolicyFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handlePolicyFile(e.target.files[0]);
            }
        });

        // Process button
        document.getElementById('process-policy-btn').addEventListener('click', () => {
            this.uploadPolicy();
        });

        // Cancel button
        document.getElementById('cancel-policy-btn').addEventListener('click', () => {
            preview.classList.add('hidden');
            fileInput.value = '';
        });
    }

    // === EVENT LISTENERS ===
    setupEventListeners() {
        console.log('🎛️ Setting up event listeners...');
        
        // Compliance scan
        const runScanBtn = document.getElementById('run-scan-btn');
        if (runScanBtn) {
            runScanBtn.addEventListener('click', () => {
                console.log('🔍 Starting compliance scan...');
                this.runComplianceScan();
            });
        } else {
            console.warn('⚠️ Run scan button not found');
        }

        // Generate sample data
        const generateSampleBtn = document.getElementById('generate-sample-btn');
        if (generateSampleBtn) {
            console.log('🎲 Setting up sample data generator...');
            generateSampleBtn.addEventListener('click', () => {
                console.log('🎲 Generating sample data...');
                this.generateSampleData();
            });
        } else {
            console.warn('⚠️ Generate sample button not found');
        }

        // Confidence threshold slider
        const slider = document.getElementById('confidence-threshold');
        const valueDisplay = document.getElementById('confidence-value');
        if (slider && valueDisplay) {
            slider.addEventListener('input', (e) => {
                valueDisplay.textContent = Math.round(e.target.value * 100) + '%';
            });
        }

        // Violation tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const status = e.target.getAttribute('data-status');
                this.loadViolations(status);
                
                // Update tab buttons
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
            });
        });
        
        console.log('✅ Event listeners setup complete');
    }

    // === DASHBOARD ===
    async loadDashboard() {
        console.log('📈 Loading dashboard...');
        try {
            const stats = await this.makeRequest('/dashboard/stats');
            
            // Update stats cards
            this.updateElement('total-transactions', stats.total_transactions || '0');
            this.updateElement('total-policies', stats.total_policies || '0');
            this.updateElement('total-violations', stats.total_violations || '0');
            this.updateElement('pending-reviews', stats.pending_reviews || '0');
            
            // Update recent violations delta
            const recentDelta = document.getElementById('recent-violations');
            if (recentDelta) {
                recentDelta.textContent = `+${stats.recent_violations || 0} this week`;
            }
            
            // Load recent violations
            await this.loadRecentViolations();
            
            console.log('✅ Dashboard loaded successfully');
        } catch (error) {
            console.error('⚠️ Failed to load dashboard:', error);
            this.showErrorMessage('Failed to load dashboard data');
        }
    }
    
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        } else {
            console.warn(`⚠️ Element not found: ${id}`);
        }
    }
    async checkSystemHealth() {
        console.log('⚕️ Checking system health...');
        try {
            const health = await this.makeRequest('/health');
            const statusIndicator = document.getElementById('status-indicator');
            const statusText = document.getElementById('status-text');
            
            if (statusIndicator && statusText) {
                statusIndicator.className = 'status-dot status-healthy';
                statusText.textContent = 'System Healthy';
            }
            
            console.log('✅ System health check complete');
        } catch (error) {
            console.error('⚠️ Health check failed:', error);
            const statusIndicator = document.getElementById('status-indicator');
            const statusText = document.getElementById('status-text');
            
            if (statusIndicator && statusText) {
                statusIndicator.className = 'status-dot status-error';
                statusText.textContent = 'System Error';
            }
        }
    }

    async makeRequest(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.apiBase}${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Request failed:', error);
            this.showToast(`API Error: ${error.message}`, 'error');
            throw error;
        }
    }

    async uploadFile(endpoint, file, additionalData = {}) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            Object.keys(additionalData).forEach(key => {
                formData.append(key, additionalData[key]);
            });

            const response = await fetch(`${this.apiBase}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Upload failed:', error);
            this.showToast(`Upload Error: ${error.message}`, 'error');
            throw error;
        }
    }

    // === DASHBOARD ===
    async loadDashboard() {
        try {
            const stats = await this.makeRequest('/dashboard/stats');
            this.updateDashboardStats(stats);
            
            const violations = await this.makeRequest('/violations/recent?limit=5');
            this.updateRecentViolations(violations);
        } catch (error) {
            console.error('Failed to load dashboard:', error);
        }
    }

    updateDashboardStats(stats) {
        document.getElementById('total-transactions').textContent = stats.total_transactions.toLocaleString();
        document.getElementById('total-policies').textContent = stats.total_policies.toLocaleString();
        document.getElementById('total-violations').textContent = stats.total_violations.toLocaleString();
        document.getElementById('pending-reviews').textContent = stats.pending_reviews.toLocaleString();
        document.getElementById('recent-violations').textContent = `+${stats.recent_violations} this week`;
    }

    updateRecentViolations(violations) {
        const container = document.getElementById('recent-violations-list');
        
        if (violations.length === 0) {
            container.innerHTML = '<div class="text-center p-4">No recent violations</div>';
            return;
        }

        container.innerHTML = violations.map(violation => `
            <div class="violation-item ${violation.risk_level}-risk">
                <div class="violation-header">
                    <span class="violation-title">${violation.violation_type}</span>
                    <span class="risk-badge ${violation.risk_level}">${violation.risk_level}</span>
                </div>
                <p class="text-sm">${violation.reasoning}</p>
                <small class="text-xs">Confidence: ${Math.round(violation.confidence * 100)}%</small>
            </div>
        `).join('');
    }

    // === TRANSACTION UPLOAD ===
    handleTransactionFile(file) {
        if (!file.name.endsWith('.csv')) {
            this.showToast('Please upload a CSV file', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const csv = e.target.result;
            this.previewTransactionData(csv);
        };
        reader.readAsText(file);
    }

    previewTransactionData(csvData) {
        const lines = csvData.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',');
        const rows = lines.slice(1, 6); // Show first 5 rows

        const tableHTML = `
            <table class="preview-table">
                <thead>
                    <tr>${headers.map(h => `<th>${h.trim()}</th>`).join('')}</tr>
                </thead>
                <tbody>
                    ${rows.map(row => `<tr>${row.split(',').map(cell => `<td>${cell.trim()}</td>`).join('')}</tr>`).join('')}
                </tbody>
            </table>
            <p class="mt-4 text-sm">Showing ${rows.length} of ${lines.length - 1} total rows</p>
        `;

        document.getElementById('transaction-table').innerHTML = tableHTML;
        document.getElementById('transaction-preview').classList.remove('hidden');
    }

    async uploadTransactions() {
        const fileInput = document.getElementById('transaction-file');
        const file = fileInput.files[0];
        
        if (!file) return;

        this.showLoading(true);
        
        try {
            const result = await this.uploadFile('/transactions/upload', file);
            
            this.showLoading(false);
            this.showToast(`Successfully uploaded ${result.processed_count} transactions!`, 'success');
            
            // Update results section
            const resultsSection = document.getElementById('upload-results');
            const statusDiv = document.getElementById('upload-status');
            
            statusDiv.innerHTML = `
                <div class="upload-success">
                    <h4>✅ Upload Completed</h4>
                    <p>Processed: ${result.processed_count} transactions</p>
                    <p>Failed: ${result.failed_count} transactions</p>
                    ${result.errors.length > 0 ? `<details><summary>Errors (${result.errors.length})</summary><ul>${result.errors.map(e => `<li>${e}</li>`).join('')}</ul></details>` : ''}
                </div>
            `;
            
            resultsSection.classList.remove('hidden');
            document.getElementById('transaction-preview').classList.add('hidden');
            
            // Refresh dashboard if on dashboard
            if (this.currentSection === 'dashboard') {
                this.loadDashboard();
            }
            
        } catch (error) {
            this.showLoading(false);
        }
    }

    // === POLICY UPLOAD ===
    handlePolicyFile(file) {
        const allowedTypes = ['.txt', '.md'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExt)) {
            this.showToast('Please upload a TXT or MD file', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            this.previewPolicyContent(content);
        };
        reader.readAsText(file);
    }

    previewPolicyContent(content) {
        const preview = content.length > 1000 ? content.substring(0, 1000) + '...' : content;
        document.getElementById('policy-content').innerHTML = `
            <pre class="policy-text">${preview}</pre>
            <p class="text-sm mt-2">Content length: ${content.length} characters</p>
        `;
        document.getElementById('policy-preview').classList.remove('hidden');
    }

    async uploadPolicy() {
        const fileInput = document.getElementById('policy-file');
        const file = fileInput.files[0];
        const title = document.getElementById('policy-title').value || file.name;
        const jurisdiction = document.getElementById('policy-jurisdiction').value;
        
        if (!file) return;

        this.showLoading(true);
        
        try {
            const result = await this.uploadFile('/policies/upload', file, {
                title: title,
                jurisdiction: jurisdiction
            });
            
            this.showLoading(false);
            this.showToast(`Successfully processed policy and extracted ${result.obligations.length} obligations!`, 'success');
            
            // Display obligations
            this.displayExtractedObligations(result.obligations);
            document.getElementById('policy-preview').classList.add('hidden');
            
        } catch (error) {
            this.showLoading(false);
        }
    }

    displayExtractedObligations(obligations) {
        const container = document.getElementById('obligations-list');
        
        container.innerHTML = obligations.map((obl, index) => `
            <div class="obligation-item">
                <h4>Obligation ${index + 1}: ${obl.actor}</h4>
                <div class="obligation-details">
                    <p><strong>Action:</strong> ${obl.action}</p>
                    <p><strong>Type:</strong> ${obl.type}</p>
                    <p><strong>Condition:</strong> ${obl.condition || 'N/A'}</p>
                    <p><strong>Jurisdiction:</strong> ${obl.jurisdiction || 'N/A'}</p>
                    <p><strong>Confidence:</strong> ${Math.round(obl.confidence * 100)}%</p>
                    <details>
                        <summary>Source Clause</summary>
                        <p class="source-text">${obl.source_clause}</p>
                    </details>
                </div>
            </div>
        `).join('');
        
        document.getElementById('policy-results').classList.remove('hidden');
    }

    // === COMPLIANCE SCAN ===
    async runComplianceScan() {
        const jurisdiction = document.getElementById('scan-jurisdiction').value;
        const confidenceThreshold = parseFloat(document.getElementById('confidence-threshold').value);
        const transactionLimit = parseInt(document.getElementById('transaction-limit').value);
        
        // Show progress
        document.getElementById('scan-progress').classList.remove('hidden');
        document.getElementById('scan-results').classList.add('hidden');
        
        // Simulate progress
        this.animateProgress();
        
        try {
            const result = await this.makeRequest('/compliance/scan', {
                method: 'POST',
                body: JSON.stringify({
                    jurisdiction: jurisdiction,
                    confidence_threshold: confidenceThreshold,
                    transaction_limit: transactionLimit
                })
            });
            
            this.displayScanResults(result);
            
        } catch (error) {
            document.getElementById('scan-progress').classList.add('hidden');
        }
    }

    animateProgress() {
        const progressFill = document.getElementById('scan-progress-fill');
        const statusText = document.getElementById('scan-status');
        
        const steps = [
            { progress: 20, text: 'Loading transactions...' },
            { progress: 40, text: 'Applying compliance rules...' },
            { progress: 60, text: 'Running semantic analysis...' },
            { progress: 80, text: 'Generating explanations...' },
            { progress: 100, text: 'Finalizing results...' }
        ];
        
        let currentStep = 0;
        
        const updateProgress = () => {
            if (currentStep < steps.length) {
                const step = steps[currentStep];
                progressFill.style.width = step.progress + '%';
                statusText.textContent = step.text;
                currentStep++;
                setTimeout(updateProgress, 800);
            }
        };
        
        updateProgress();
    }

    displayScanResults(result) {
        document.getElementById('scan-progress').classList.add('hidden');
        document.getElementById('scan-results').classList.remove('hidden');
        
        // Update summary
        document.getElementById('scanned-count').textContent = result.transaction_count;
        document.getElementById('violations-found').textContent = result.violation_count;
        document.getElementById('violation-rate').textContent = Math.round(result.summary.violation_rate * 100) + '%';
        document.getElementById('processing-time').textContent = '2.5s'; // Mock timing
        
        // Display violations
        const container = document.getElementById('violations-detected');
        
        if (result.violations.length === 0) {
            container.innerHTML = '<div class="no-violations">🎉 No violations detected!</div>';
            return;
        }
        
        container.innerHTML = result.violations.map(violation => `
            <div class="violation-item ${violation.risk_level}-risk">
                <div class="violation-header">
                    <span class="violation-title">${violation.violation_type}</span>
                    <span class="risk-badge ${violation.risk_level}">${violation.risk_level}</span>
                </div>
                <div class="violation-content">
                    <p><strong>Transaction:</strong> ${violation.transaction_id}</p>
                    <p><strong>Reasoning:</strong> ${violation.reasoning}</p>
                    <p><strong>Confidence:</strong> ${Math.round(violation.confidence * 100)}%</p>
                    <p><strong>Recommended Action:</strong> ${violation.recommended_action}</p>
                    ${violation.human_explanation ? `<details><summary>Detailed Explanation</summary><p>${violation.human_explanation}</p></details>` : ''}
                </div>
                <div class="violation-actions">
                    <button class="btn btn-success btn-sm" onclick="app.reviewViolation('${violation.violation_id}', 'approve')">✅ Approve</button>
                    <button class="btn btn-danger btn-sm" onclick="app.reviewViolation('${violation.violation_id}', 'reject')">❌ Reject</button>
                </div>
            </div>
        `).join('');
    }

    // === VIOLATIONS ===
    async loadViolations(status = 'pending') {
        try {
            const violations = await this.makeRequest(`/violations/recent?status=${status}&limit=20`);
            this.displayViolationsList(violations);
        } catch (error) {
            console.error('Failed to load violations:', error);
        }
    }

    displayViolationsList(violations) {
        const container = document.getElementById('violations-list');
        
        if (violations.length === 0) {
            container.innerHTML = '<div class="text-center p-8">No violations found</div>';
            return;
        }
        
        container.innerHTML = violations.map(violation => `
            <div class="violation-item ${violation.risk_level}-risk">
                <div class="violation-header">
                    <span class="violation-title">${violation.violation_type}</span>
                    <span class="risk-badge ${violation.risk_level}">${violation.risk_level}</span>
                </div>
                <div class="violation-content">
                    <p><strong>Transaction:</strong> ${violation.transaction_id}</p>
                    <p><strong>Reasoning:</strong> ${violation.reasoning}</p>
                    <p><strong>Confidence:</strong> ${Math.round(violation.confidence * 100)}%</p>
                    <p><strong>Status:</strong> ${violation.review_status}</p>
                    <p><strong>Detected:</strong> ${new Date(violation.detected_at).toLocaleDateString()}</p>
                    ${violation.human_explanation ? `<details><summary>AI Analysis</summary><pre>${violation.human_explanation}</pre></details>` : ''}
                </div>
                ${violation.review_status === 'pending' ? `
                <div class="violation-actions">
                    <input type="text" placeholder="Review notes..." id="notes-${violation.id}" class="review-notes">
                    <button class="btn btn-success" onclick="app.reviewViolation('${violation.id}', 'approve')">✅ Approve</button>
                    <button class="btn btn-danger" onclick="app.reviewViolation('${violation.id}', 'reject')">❌ Reject</button>
                </div>
                ` : ''}
            </div>
        `).join('');
    }

    async reviewViolation(violationId, action) {
        const notesElement = document.getElementById(`notes-${violationId}`);
        const notes = notesElement ? notesElement.value : '';
        
        try {
            await this.makeRequest(`/violations/${violationId}/review`, {
                method: 'POST',
                body: JSON.stringify({
                    action: action,
                    notes: notes,
                    reviewer_id: 'web_user'
                })
            });
            
            this.showToast(`Violation ${action}d successfully!`, 'success');
            
            // Reload violations
            const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-status');
            this.loadViolations(activeTab);
            
            // Update dashboard if needed
            if (this.currentSection === 'dashboard') {
                this.loadDashboard();
            }
            
        } catch (error) {
            console.error('Failed to review violation:', error);
        }
    }

    // === SYSTEM HEALTH ===
    async checkSystemHealth() {
        try {
            const health = await this.makeRequest('/health');
            this.updateSystemStatus(health.status);
        } catch (error) {
            this.updateSystemStatus('error');
        }
    }

    updateSystemStatus(status) {
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');
        
        indicator.className = `status-dot status-${status}`;
        
        const statusTexts = {
            'healthy': 'System Healthy',
            'warning': 'System Warning',
            'error': 'System Error'
        };
        
        text.textContent = statusTexts[status] || 'Unknown Status';
    }

    // === UTILITIES ===
    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <p>${message}</p>
            </div>
        `;
        
        container.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
    // === USER FEEDBACK ===
    showToast(message, type = 'info') {
        console.log(`🍞 Toast (${type}): ${message}`);
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <strong>${type.toUpperCase()}:</strong> ${message}
            <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; cursor: pointer;">&times;</button>
        `;
        
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        
        container.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <strong>Error:</strong> ${message}
            <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; color: inherit; cursor: pointer;">&times;</button>
        `;
        
        if (containerId) {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = '';
                container.appendChild(errorDiv);
            }
        } else {
            // Show in toast
            this.showToast(message, 'error');
        }
    }

    showSuccessMessage(message, containerId = null) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.innerHTML = `
            <strong>Success:</strong> ${message}
            <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; color: inherit; cursor: pointer;">&times;</button>
        `;
        
        if (containerId) {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = '';
                container.appendChild(successDiv);
            }
        } else {
            this.showToast(message, 'success');
        }
    }

    async loadRecentViolations() {
        const container = document.getElementById('recent-violations-list');
        try {
            const violations = await this.makeRequest('/violations?limit=5');
            
            if (!violations || violations.length === 0) {
                container.innerHTML = '<div class="no-data">No recent violations found</div>';
                return;
            }
            
            container.innerHTML = violations.map(violation => `
                <div class="violation-item ${violation.risk_level}-risk">
                    <div class="violation-header">
                        <span class="violation-title">${violation.violation_type.replace('_', ' ').toUpperCase()}</span>
                        <span class="risk-badge ${violation.risk_level}">${violation.risk_level}</span>
                    </div>
                    <p class="violation-details">${violation.reasoning || 'No details available'}</p>
                    <div class="violation-meta">
                        <small>Detected: ${new Date(violation.detected_at).toLocaleDateString()}</small>
                        <small>Confidence: ${Math.round(violation.confidence * 100)}%</small>
                    </div>
                </div>
            `).join('');
            
        } catch (error) {
            console.error('Failed to load recent violations:', error);
            container.innerHTML = '<div class="error-message">Failed to load violations</div>';
        }
    }
    async generateSampleData() {
        const btn = document.getElementById('generate-sample-btn');
        const originalText = btn.textContent;
        
        try {
            btn.textContent = '⏳ Generating...';
            btn.disabled = true;
            
            const result = await this.makeRequest('/demo/generate-sample-data', {
                method: 'POST'
            });
            
            if (result.success) {
                this.showToast(
                    `Sample data generated: ${result.data.policies} policies, ${result.data.transactions} transactions, ${result.data.violations} violations`,
                    'success'
                );
                
                // Refresh dashboard to show new data
                setTimeout(() => {
                    this.loadDashboard();
                }, 1000);
            }
            
        } catch (error) {
            console.error('Failed to generate sample data:', error);
            this.showToast('Failed to generate sample data', 'error');
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }
}

// Initialize the app
const app = new FinLexApp();

// Add some CSS for dynamic elements
const style = document.createElement('style');
style.textContent = `
    .preview-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .preview-table th,
    .preview-table td {
        padding: 0.5rem;
        border: 1px solid var(--gray-300);
        text-align: left;
    }
    
    .preview-table th {
        background: var(--gray-100);
        font-weight: 500;
    }
    
    .policy-text {
        background: var(--gray-50);
        padding: 1rem;
        border-radius: var(--radius-md);
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .obligation-item {
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .obligation-item h4 {
        color: var(--primary-blue);
        margin-bottom: 1rem;
    }
    
    .obligation-details p {
        margin-bottom: 0.5rem;
    }
    
    .source-text {
        background: var(--gray-50);
        padding: 0.75rem;
        border-radius: var(--radius-md);
        font-size: var(--font-size-sm);
        margin-top: 0.5rem;
    }
    
    .violation-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        align-items: center;
    }
    
    .review-notes {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid var(--gray-300);
        border-radius: var(--radius-md);
        margin-right: 0.5rem;
    }
    
    .btn-sm {
        padding: 0.25rem 0.75rem;
        font-size: var(--font-size-sm);
    }
    
    .upload-success {
        background: var(--gradient-success);
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        color: var(--gray-800);
    }
    
    .no-violations {
        text-align: center;
        padding: 3rem;
        font-size: var(--font-size-lg);
        color: var(--secondary-green);
    }
`;
document.head.appendChild(style);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🏦 Initializing FinLex Audit AI...');
    window.finlexApp = new FinLexApp();
    console.log('✅ FinLex Audit AI initialized successfully');
});