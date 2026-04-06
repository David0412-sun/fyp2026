/**
 * 8-Channel Microphone Array Localization System
 * Frontend JavaScript Application
 * 
 * Handles WebSocket communication, real-time visualization,
 * and user interface interactions.
 */

(function() {
  'use strict';

  // =============================================================================
  // Configuration
  // =============================================================================

  const CONFIG = {
    socketPath: '/socket.io',
    socketTransports: ['websocket', 'polling'],
    reconnectAttempts: 5,
    reconnectDelay: 1000,
    maxHistoryPoints: 200,
    chartUpdateInterval: 100, // ms
    polarChartRotation: 0
  };

  // =============================================================================
  // Helpers
  // =============================================================================

  function wrap360(deg) {
    // Normalize to [0, 360)
    deg = Number(deg);
    if (!isFinite(deg)) return 0;
    deg = deg % 360;
    return deg < 0 ? deg + 360 : deg;
  }

  function angleStdToDisplay(degStd) {
    // Backend uses standard atan2 angle: 0°=right, CCW positive (wrap180).
    // We display standard math convention: 0°=right, CCW positive in [0,360).
    return wrap360(degStd);
  }

  function formatWallTime(iso) {
    if (!iso) return '--';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    return d.toLocaleString(undefined, { hour12: false });
  }

// =============================================================================
  // State Management
  // =============================================================================

  const state = {
    connected: false,
    engineRunning: false,
    socket: null,
    data: {
      angle: null,
      angleRaw: null,
      distance: null,
      snr: null,
      coherence: null,
      diameter: null,
      timestamp: 0,
      wallTimeIso: null,
      valid: false,
      updateFlag: false,
      errorPlane: null,
      errorNear: null,
      rhoHat: null
    },
    history: {
      time: [],
      angle: [],
      distance: []
    },
    device: {
      name: null,
      channels: null,
      samplerate: null
    }
  };

  // =============================================================================
  // DOM Elements Cache
  // =============================================================================

  const elements = {};

  /**
   * Initialize element references
   */
  function initElements() {
    // Status indicators
    elements.connectionStatus = document.getElementById('connectionStatus');
    elements.engineStatus = document.getElementById('engineStatus');
    
    // Display values
    elements.angleValue = document.getElementById('angleValue');
    elements.distanceValue = document.getElementById('distanceValue');
    elements.timeValue = document.getElementById('timeValue');
    elements.diameterValue = document.getElementById('diameterValue');
    elements.channelsValue = document.getElementById('channelsValue');
    elements.sampleRateValue = document.getElementById('sampleRateValue');
    elements.snrValue = document.getElementById('snrValue');
    elements.coherenceValue = document.getElementById('coherenceValue');
    elements.planeError = document.getElementById('planeError');
    elements.nearError = document.getElementById('nearError');
    elements.rhoHat = document.getElementById('rhoHat');
    elements.frameCount = document.getElementById('frameCount');
    elements.deviceName = document.getElementById('deviceName');
    elements.deviceStatus = document.getElementById('deviceStatus');
    
    // Meters
    elements.snrMeter = document.getElementById('snrMeter');
    elements.coherenceMeter = document.getElementById('coherenceMeter');
    
    // Indicators
    elements.detectionIndicator = document.getElementById('detectionIndicator');
    elements.updateIndicator = document.getElementById('updateIndicator');
    
    // Buttons
    elements.startEngine = document.getElementById('startEngine');
    elements.stopEngine = document.getElementById('stopEngine');
    elements.clearCharts = document.getElementById('clearCharts');
    elements.updateDiameter = document.getElementById('updateDiameter');
    elements.applyParams = document.getElementById('applyParams');
    
    // Inputs
    elements.diameterInput = document.getElementById('diameterInput');
    elements.cSoundInput = document.getElementById('cSoundInput');
    elements.cohThInput = document.getElementById('cohThInput');
    elements.vadThInput = document.getElementById('vadThInput');
    
    // Loading overlay
    elements.loadingOverlay = document.getElementById('loadingOverlay');
    
    // Toast container
    elements.toastContainer = document.getElementById('toastContainer');
  }

  // =============================================================================
  // Chart Instances
  // =============================================================================

  const charts = {
    polar: null,
    angle: null,
    distance: null
  };

  // =============================================================================
  // Toast Notifications
  // =============================================================================

  /**
   * Show a toast notification
   * @param {string} message - Notification message
   * @param {string} type - Type: 'success', 'error', 'warning', 'info'
   * @param {number} duration - Display duration in ms (0 = no auto-hide)
   */
  function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
      success: 'fa-check-circle',
      error: 'fa-exclamation-circle',
      warning: 'fa-exclamation-triangle',
      info: 'fa-info-circle'
    };
    
    toast.innerHTML = `
      <i class="fas ${icons[type]}"></i>
      <span class="toast-message">${message}</span>
      <button class="toast-close"><i class="fas fa-times"></i></button>
    `;
    
    // Close button handler
    toast.querySelector('.toast-close').addEventListener('click', () => {
      hideToast(toast);
    });
    
    elements.toastContainer.appendChild(toast);
    
    // Auto-hide
    if (duration > 0) {
      setTimeout(() => hideToast(toast), duration);
    }
    
    return toast;
  }

  /**
   * Hide a toast notification
   * @param {HTMLElement} toast - Toast element to hide
   */
  function hideToast(toast) {
    if (!toast) return;
    toast.classList.add('hiding');
    setTimeout(() => toast.remove(), 300);
  }

  // =============================================================================
  // Chart Initialization
  // =============================================================================

  /**
   * Initialize the polar plot chart
   */
  function initPolarChart() {
    const ctx = document.getElementById('polarChart');
    if (!ctx) return;
    
    charts.polar = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['90°', '45°', '0°', '315°', '270°', '225°', '180°', '135°'],
        datasets: [{
          label: 'Microphone Array',
          data: [1, 1, 1, 1, 1, 1, 1, 1],
          backgroundColor: 'rgba(59, 130, 246, 0.2)',
          borderColor: 'rgba(59, 130, 246, 0.8)',
          borderWidth: 2,
          pointRadius: 6,
          pointBackgroundColor: '#3b82f6'
        }, {
          label: 'Source Direction',
          data: [0, 0, 0, 0, 0, 0, 0, 0],
          backgroundColor: 'rgba(245, 158, 11, 0.3)',
          borderColor: '#f59e0b',
          borderWidth: 3,
          pointRadius: 0,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          r: {
            beginAtZero: true,
            max: 1.2,
            ticks: {
              display: false
            },
            grid: {
              color: 'rgba(51, 65, 85, 0.5)'
            },
            angleLines: {
              color: 'rgba(51, 65, 85, 0.5)'
            },
            pointLabels: {
              color: '#94a3b8',
              font: {
                size: 11
              }
            }
          }
        },
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            enabled: false
          }
        },
        animation: {
          duration: 150
        }
      }
    });
  }

  /**
   * Initialize the angle time series chart
   */
  function initAngleChart() {
    const ctx = document.getElementById('angleChart');
    if (!ctx) return;
    
    charts.angle = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Angle (°)',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            display: false,
            grid: {
              display: false
            }
          },
          y: {
            min: 0,
            max: 360,
            grid: {
              color: 'rgba(51, 65, 85, 0.3)'
            },
            ticks: {
              color: '#94a3b8',
              font: {
                size: 10
              }
            }
          }
        },
        plugins: {
          legend: {
            display: false
          }
        },
        animation: {
          duration: 100
        }
      }
    });
  }

  /**
   * Initialize the distance time series chart
   */
  function initDistanceChart() {
    const ctx = document.getElementById('distanceChart');
    if (!ctx) return;
    
    charts.distance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Distance (m)',
          data: [],
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            display: false,
            grid: {
              display: false
            }
          },
          y: {
            min: 0,
            max: 2,
            grid: {
              color: 'rgba(51, 65, 85, 0.3)'
            },
            ticks: {
              color: '#94a3b8',
              font: {
                size: 10
              }
            }
          }
        },
        plugins: {
          legend: {
            display: false
          }
        },
        animation: {
          duration: 100
        }
      }
    });
  }

  /**
   * Initialize all charts
   */
  function initCharts() {
    initPolarChart();
    initAngleChart();
    initDistanceChart();
  }

  // =============================================================================
  // Display Updates
  // =============================================================================

  /**
   * Update the polar chart with new direction
   * @param {number} angle - Angle in degrees
   */
  function updatePolarChart(angle) {
    if (!charts.polar || angle === null || !isFinite(angle)) return;

    // Chart.js radar angles advance CLOCKWISE starting from the top (0°=up, clockwise).
    // We display angles in standard math convention:
    //   0° = right, 90° = up, 180° = left, 270° = down (range 0..360, CCW).
    // Convert standard angle to chart angle (0°=up, clockwise): angleChart = 90° - angle.
    const angleChart = wrap360(90 - angle);

    // Bin centers follow the chart label order (top -> clockwise).
    const binAngles = [0, 45, 90, 135, 180, 225, 270, 315];

    // Circular difference helper
    const circDiff = (a, b) => {
      let d = Math.abs(a - b) % 360;
      return d > 180 ? 360 - d : d;
    };

    const direction = [];
    for (let i = 0; i < 8; i++) {
      const diff = circDiff(angleChart, binAngles[i]);
      if (diff < 45) {
        direction.push(1 - (diff / 45));
      } else {
        direction.push(0);
      }
    }

    charts.polar.data.datasets[1].data = direction;
    charts.polar.update('none');
  }
  /**
   * Update time series charts
   */
  function updateTimeCharts() {
    const maxPoints = CONFIG.maxHistoryPoints;
    
    // Update angle chart
    if (charts.angle) {
      const angleLabels = state.history.time.slice(-maxPoints).map(t => t.toFixed(1));
      const angleData = state.history.angle.slice(-maxPoints);
      
      charts.angle.data.labels = angleLabels;
      charts.angle.data.datasets[0].data = angleData;
      charts.angle.update('none');
    }
    
    // Update distance chart
    if (charts.distance) {
      const distLabels = state.history.time.slice(-maxPoints).map(t => t.toFixed(1));
      const distData = state.history.distance.slice(-maxPoints);
      
      charts.distance.data.labels = distLabels;
      charts.distance.data.datasets[0].data = distData;
      charts.distance.update('none');
    }
  }

  /**
   * Update all display elements with current data
   */
  function updateDisplay() {
    // Main values
    if (elements.angleValue) {
      elements.angleValue.textContent = state.data.angle !== null 
        ? state.data.angle.toFixed(2) : '--';
    }
    
    if (elements.distanceValue) {
      elements.distanceValue.textContent = state.data.distance !== null 
        ? state.data.distance.toFixed(3) : '--';
    }

    if (elements.timeValue) {
      elements.timeValue.textContent = formatWallTime(state.data.wallTimeIso);
    }

    if (elements.timeValue) {
      elements.timeValue.textContent = formatWallTime(state.data.wallTimeIso);
    }
    
    if (elements.diameterValue) {
      elements.diameterValue.textContent = state.data.diameter !== null 
        ? state.data.diameter.toFixed(3) + ' m' : '--';
    }
    
    // Device info
    if (elements.channelsValue && state.device.channels) {
      elements.channelsValue.textContent = state.device.channels;
    }
    
    if (elements.sampleRateValue && state.device.samplerate) {
      elements.sampleRateValue.textContent = state.device.samplerate.toFixed(0);
    }
    
    if (elements.deviceName && state.device.name) {
      elements.deviceName.textContent = state.device.name;
    }
    
    // Signal quality
    if (elements.snrValue && state.data.snr !== null) {
      elements.snrValue.textContent = state.data.snr.toFixed(1) + ' dB';
      elements.snrMeter.style.width = Math.min(100, Math.max(0, (state.data.snr + 20) * 2)) + '%';
    }
    
    if (elements.coherenceValue && state.data.coherence !== null) {
      elements.coherenceValue.textContent = state.data.coherence.toFixed(2);
      elements.coherenceMeter.style.width = Math.min(100, state.data.coherence * 50) + '%';
    }
    
    // Error metrics
    if (elements.planeError && state.data.errorPlane !== null) {
      elements.planeError.textContent = state.data.errorPlane.toFixed(1) + ' us';
    }
    
    if (elements.nearError && state.data.errorNear !== null) {
      elements.nearError.textContent = state.data.errorNear !== null 
        ? state.data.errorNear.toFixed(1) + ' us' : '-- us';
    }
    
    if (elements.rhoHat && state.data.rhoHat !== null) {
      elements.rhoHat.textContent = state.data.rhoHat !== null 
        ? state.data.rhoHat.toFixed(3) + ' m' : '-- m';
    }
    
    // Indicators
    if (state.data.valid) {
      elements.detectionIndicator.classList.add('active');
    } else {
      elements.detectionIndicator.classList.remove('active');
    }
    
    if (state.data.updateFlag) {
      elements.updateIndicator.classList.add('active');
    } else {
      elements.updateIndicator.classList.remove('active');
    }
    
    // Update polar chart
    if (state.data.angle !== null) {
      updatePolarChart(state.data.angle);
    }
  }

  /**
   * Update engine status display
   * @param {boolean} running - Whether engine is running
   */
  function updateEngineStatus(running) {
    state.engineRunning = running;
    
    const statusDot = elements.engineStatus.querySelector('.status-dot');
    const statusText = elements.engineStatus.querySelector('.status-text');
    
    if (running) {
      statusDot.classList.remove('inactive');
      statusDot.classList.add('active');
      statusText.textContent = 'Engine Running';
      
      elements.startEngine.disabled = true;
      elements.stopEngine.disabled = false;
    } else {
      statusDot.classList.remove('active');
      statusDot.classList.add('inactive');
      statusText.textContent = 'Engine Stopped';
      
      elements.startEngine.disabled = false;
      elements.stopEngine.disabled = true;
    }
  }

  // =============================================================================
  // WebSocket Communication
  // =============================================================================

  /**
   * Initialize WebSocket connection
   */
  function initSocket() {
    const socketUrl = window.location.origin;
    
    state.socket = io(socketUrl, {
      path: CONFIG.socketPath,
      transports: CONFIG.socketTransports,
      reconnectionAttempts: CONFIG.reconnectAttempts,
      reconnectionDelay: CONFIG.reconnectDelay
    });
    
    // Connection events
    state.socket.on('connect', () => {
      console.log('WebSocket connected');
      state.connected = true;
      
      const statusDot = elements.connectionStatus.querySelector('.status-dot');
      const statusText = elements.connectionStatus.querySelector('.status-text');
      
      statusDot.classList.remove('disconnected', 'connecting');
      statusDot.classList.add('connected');
      statusText.textContent = 'Connected';
      
      showToast('Connected to server', 'success', 3000);
    });
    
    state.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      state.connected = false;
      
      const statusDot = elements.connectionStatus.querySelector('.status-dot');
      const statusText = elements.connectionStatus.querySelector('.status-text');
      
      statusDot.classList.remove('connected');
      statusDot.classList.add('disconnected');
      statusText.textContent = 'Disconnected';
      
      showToast('Disconnected from server: ' + reason, 'warning');
      
      updateEngineStatus(false);
    });
    
    state.socket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      state.connected = false;
      
      const statusDot = elements.connectionStatus.querySelector('.status-dot');
      statusDot.classList.remove('connecting');
      statusDot.classList.add('disconnected');
      
      showToast('Connection failed: ' + error.message, 'error');
    });
    
    // Data events
    state.socket.on('localization_data', handleLocalizationData);
    
    state.socket.on('engine_status', handleEngineStatus);
    
    state.socket.on('config_update', handleConfigUpdate);
    
    state.socket.on('error', handleError);
  }

  /**
   * Handle localization data from server
   * @param {Object} data - Localization data
   */
  function handleLocalizationData(data) {
    if (data.type !== 'data') return;
    
    // Update state
    state.data.angleRaw = data.angle;
    state.data.angle = (data.angle !== null && isFinite(data.angle)) ? angleStdToDisplay(data.angle) : null;
    state.data.distance = data.distance;
    state.data.snr = data.snr_db;
    state.data.coherence = data.coherence;
    state.data.diameter = data.diameter_m;
    state.data.timestamp = data.timestamp;
    state.data.wallTimeIso = data.wall_time_iso || null;
    state.data.valid = data.valid;
    state.data.updateFlag = data.update_flag;
    state.data.errorPlane = data.error_plane_us;
    state.data.errorNear = data.error_near_us;
    state.data.rhoHat = data.rho_hat;
    
    // Update history
    state.history.time.push(data.timestamp);
    state.history.angle.push(data.angle);
    state.history.distance.push(data.distance);
    
    // Trim history if needed
    const maxLen = CONFIG.maxHistoryPoints * 2;
    if (state.history.time.length > maxLen) {
      state.history.time = state.history.time.slice(-maxLen);
      state.history.angle = state.history.angle.slice(-maxLen);
      state.history.distance = state.history.distance.slice(-maxLen);
    }
    
    // Update display
    updateDisplay();
  }

  /**
   * Handle engine status updates
   * @param {Object} status - Engine status
   */
  function handleEngineStatus(status) {
    if (status.type !== 'status') return;
    
    updateEngineStatus(status.running);
    
    if (status.device_info) {
      state.device.name = status.device_info.name;
      state.device.channels = status.device_info.channels;
      state.device.samplerate = status.device_info.samplerate;
      
      // Update all device info display elements
      if (elements.deviceName) {
        elements.deviceName.textContent = status.device_info.name;
      }
      if (elements.channelsValue) {
        elements.channelsValue.textContent = status.device_info.channels;
      }
      if (elements.sampleRateValue) {
        elements.sampleRateValue.textContent = status.device_info.samplerate.toFixed(0);
      }
      if (elements.deviceStatus) {
        elements.deviceStatus.textContent = 'Active';
      }
    }
    
    if (status.frame_count && elements.frameCount) {
      elements.frameCount.textContent = status.frame_count.toLocaleString();
    }
  }

  /**
   * Handle configuration updates from server
   * @param {Object} config - Configuration update
   */
  function handleConfigUpdate(config) {
    if (config.type !== 'config') return;
    
    showToast(`${config.parameter} updated to ${config.value}`, 'success', 3000);
  }

  /**
   * Handle error messages from server
   * @param {Object} error - Error data
   */
  function handleError(error) {
    if (error.type !== 'error') return;
    
    console.error('Server error:', error.message);
    showToast('Server error: ' + error.message, 'error');
  }

  // =============================================================================
  // API Calls
  // =============================================================================

  /**
   * Fetch current configuration
   */
  async function fetchConfig() {
    try {
      const response = await fetch('/api/config');
      const config = await response.json();
      
      if (config.algorithm) {
        if (elements.cSoundInput) {
          elements.cSoundInput.value = config.algorithm.c_sound;
        }
        if (elements.cohThInput) {
          elements.cohThInput.value = config.algorithm.coh_th;
        }
        if (elements.vadThInput) {
          elements.vadThInput.value = config.algorithm.vad_th;
        }
      }
      
      // Update device info from config if available
      if (config.device) {
        if (config.device.channels && elements.channelsValue) {
          state.device.channels = config.device.channels;
          elements.channelsValue.textContent = config.device.channels;
        }
        if (config.device.samplerate && elements.sampleRateValue) {
          state.device.samplerate = config.device.samplerate;
          elements.sampleRateValue.textContent = config.device.samplerate.toFixed(0);
        }
        if (config.device.dia_init_meters && elements.diameterInput) {
          elements.diameterInput.value = config.device.dia_init_meters;
        }
      }
      
      // Update device status to "Ready"
      if (elements.deviceStatus) {
        elements.deviceStatus.textContent = 'Ready';
      }
      
      return config;
    } catch (error) {
      console.error('Failed to fetch config:', error);
      return null;
    }
  }

  /**
   * Start the localization engine
   */
  async function startEngine() {
    if (elements.loadingOverlay) {
      elements.loadingOverlay.classList.add('active');
    }
    
    try {
      const response = await fetch('/api/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });
      
      const result = await response.json();
      
      if (result.success) {
        state.device.name = result.device.name;
        state.device.channels = result.device.channels;
        state.device.samplerate = result.device.samplerate;
        
        // Update all device info display elements immediately
        if (elements.deviceName) {
          elements.deviceName.textContent = result.device.name;
        }
        if (elements.channelsValue) {
          elements.channelsValue.textContent = result.device.channels;
        }
        if (elements.sampleRateValue) {
          elements.sampleRateValue.textContent = result.device.samplerate.toFixed(0);
        }
        if (elements.deviceStatus) {
          elements.deviceStatus.textContent = 'Active';
        }
        
        // Update display
        updateDisplay();
        
        showToast('Engine started successfully', 'success', 3000);
        
        // Request status update via WebSocket
        state.socket.emit('request_status');
      } else {
        showToast('Failed to start engine: ' + result.error, 'error');
      }
    } catch (error) {
      console.error('Failed to start engine:', error);
      showToast('Failed to start engine: ' + error.message, 'error');
    } finally {
      if (elements.loadingOverlay) {
        elements.loadingOverlay.classList.remove('active');
      }
    }
  }

  /**
   * Stop the localization engine
   */
  async function stopEngine() {
    try {
      const response = await fetch('/api/stop', {
        method: 'POST'
      });
      
      const result = await response.json();
      
      if (result.success) {
        showToast('Engine stopped', 'info');
      } else {
        showToast('Failed to stop engine: ' + result.error, 'error');
      }
    } catch (error) {
      console.error('Failed to stop engine:', error);
      showToast('Failed to stop engine: ' + error.message, 'error');
    }
  }

  /**
   * Update diameter value
   * @param {number} diameter - New diameter in meters
   */
  async function updateDiameter(diameter) {
    try {
      const response = await fetch('/api/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ diameter: diameter })
      });
      
      const result = await response.json();
      
      if (result.success) {
        state.data.diameter = diameter;
        showToast('Diameter updated', 'success');
      } else {
        showToast('Failed to update diameter: ' + result.error, 'error');
      }
    } catch (error) {
      console.error('Failed to update diameter:', error);
      showToast('Failed to update diameter: ' + error.message, 'error');
    }
  }

  /**
   * Apply all parameter changes
   */
  async function applyParameters() {
    const params = {};
    
    if (elements.cSoundInput) {
      params.c_sound = parseFloat(elements.cSoundInput.value);
    }
    if (elements.cohThInput) {
      params.coh_th = parseFloat(elements.cohThInput.value);
    }
    if (elements.vadThInput) {
      params.vad_th = parseFloat(elements.vadThInput.value);
    }
    
    try {
      const response = await fetch('/api/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
      });
      
      const result = await response.json();
      
      if (result.success) {
        showToast('Parameters updated', 'success');
      } else {
        showToast('Failed to update parameters: ' + result.error, 'error');
      }
    } catch (error) {
      console.error('Failed to update parameters:', error);
      showToast('Failed to update parameters: ' + error.message, 'error');
    }
  }

  /**
   * Clear chart history
   */
  function clearHistory() {
    state.history.time = [];
    state.history.angle = [];
    state.history.distance = [];
    
    if (charts.angle) {
      charts.angle.data.labels = [];
      charts.angle.data.datasets[0].data = [];
      charts.angle.update();
    }
    
    if (charts.distance) {
      charts.distance.data.labels = [];
      charts.distance.data.datasets[0].data = [];
      charts.distance.update();
    }
    
    showToast('Charts cleared', 'info', 2000);
  }

  // =============================================================================
  // Event Handlers
  // =============================================================================

  /**
   * Initialize event listeners
   */
  function initEventListeners() {
    // Engine control buttons
    if (elements.startEngine) {
      elements.startEngine.addEventListener('click', startEngine);
    }
    
    if (elements.stopEngine) {
      elements.stopEngine.addEventListener('click', stopEngine);
    }
    
    // Clear charts button
    if (elements.clearCharts) {
      elements.clearCharts.addEventListener('click', clearHistory);
    }
    
    // Diameter update
    if (elements.updateDiameter && elements.diameterInput) {
      elements.updateDiameter.addEventListener('click', () => {
        const diameter = parseFloat(elements.diameterInput.value);
        if (!isNaN(diameter) && diameter > 0) {
          updateDiameter(diameter);
        } else {
          showToast('Invalid diameter value', 'warning');
        }
      });
      
      // Also allow Enter key
      elements.diameterInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          elements.updateDiameter.click();
        }
      });
    }
    
    // Apply parameters
    if (elements.applyParams) {
      elements.applyParams.addEventListener('click', applyParameters);
    }
    
    // Window resize handler
    window.addEventListener('resize', () => {
      // Charts will automatically resize due to responsive: true
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + S to start engine
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        if (!state.engineRunning) {
          startEngine();
        }
      }
      
      // Ctrl/Cmd + X to stop engine
      if ((e.ctrlKey || e.metaKey) && e.key === 'x') {
        e.preventDefault();
        if (state.engineRunning) {
          stopEngine();
        }
      }
      
      // Escape to clear charts
      if (e.key === 'Escape') {
        clearHistory();
      }
    });
  }

  // =============================================================================
  // Initialization
  // =============================================================================

  /**
   * Initialize the application
   */
  async function init() {
    console.log('Initializing 8-Channel Localization Interface...');
    
    // Initialize DOM element references
    initElements();
    
    // Initialize charts
    initCharts();
    
    // Fetch initial configuration
    await fetchConfig();
    
    // Initialize WebSocket connection
    initSocket();
    
    // Initialize event listeners
    initEventListeners();
    
    // Set up periodic chart updates
    setInterval(updateTimeCharts, CONFIG.chartUpdateInterval);
    
    console.log('Initialization complete');
    
    // Show welcome toast
    setTimeout(() => {
      showToast('Ready. Press Start Engine to begin localization.', 'info', 5000);
    }, 500);
  }

  /**
   * Cleanup on page unload
   */
  function cleanup() {
    // Close WebSocket connection
    if (state.socket) {
      state.socket.disconnect();
    }
    
    // Destroy charts
    Object.values(charts).forEach(chart => {
      if (chart) {
        chart.destroy();
      }
    });
  }

  // Run initialization when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  // Cleanup on page unload
  window.addEventListener('beforeunload', cleanup);

})();
