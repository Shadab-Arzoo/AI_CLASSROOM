/**
 * ExamGuard - Native Desktop Electron Application
 * Bundles Python FastAPI AI Engine + Modern Dark Dashboard
 * Features:
 *   - Automatic Python backend lifecycle management (start/stop)
 *   - Native frameless/dark window shell
 *   - Auto-updates via GitHub Releases with confirmation prompt
 */

const { app, BrowserWindow, dialog, Notification } = require('electron');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');
const { autoUpdater } = require('electron-updater');

let mainWindow = null;
let pythonProcess = null;
const BACKEND_PORT = 8000;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;

// Configure Auto-Updater
autoUpdater.autoDownload = false;
autoUpdater.autoInstallOnAppQuit = true;

/**
 * Finds the Python interpreter executable.
 */
function getPythonPath() {
    const rootDir = path.join(__dirname, '..');
    const venvPython = path.join(rootDir, 'venv', 'bin', 'python');
    const venvWindows = path.join(rootDir, 'venv', 'Scripts', 'python.exe');

    const fs = require('fs');
    if (fs.existsSync(venvPython)) return venvPython;
    if (fs.existsSync(venvWindows)) return venvWindows;

    return process.platform === 'win32' ? 'python' : 'python3';
}

/**
 * Spawns the Python FastAPI / YOLO engine as a background child process.
 */
function startPythonBackend() {
    const rootDir = path.join(__dirname, '..');
    const pythonExe = getPythonPath();
    const scriptPath = path.join(rootDir, 'run.py');

    console.log(`[ExamGuard Desktop] Spawning Python backend via: ${pythonExe}`);
    console.log(`[ExamGuard Desktop] Script: ${scriptPath}`);

    pythonProcess = spawn(pythonExe, [scriptPath, '--no-browser'], {
        cwd: rootDir,
        env: {
            ...process.env,
            PYTHONUNBUFFERED: '1'
        },
        stdio: ['ignore', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on('data', (data) => {
        const text = data.toString().trim();
        if (text) console.log(`[Python] ${text}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        const text = data.toString().trim();
        if (text) console.error(`[Python Err] ${text}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`[ExamGuard Desktop] Python backend exited with code ${code}`);
        pythonProcess = null;
    });
}

/**
 * Polls the Python backend until it responds 200 OK.
 */
function waitForBackend(callback, retries = 30) {
    if (retries <= 0) {
        dialog.showErrorBox(
            'Backend Startup Error',
            'Failed to connect to the ExamGuard AI engine. Please ensure Python dependencies are installed.'
        );
        return;
    }

    http.get(`${BACKEND_URL}/api/status`, (res) => {
        if (res.statusCode === 200) {
            console.log('[ExamGuard Desktop] Backend is ready!');
            callback();
        } else {
            setTimeout(() => waitForBackend(callback, retries - 1), 600);
        }
    }).on('error', () => {
        setTimeout(() => waitForBackend(callback, retries - 1), 600);
    });
}

/**
 * Creates the primary application window.
 */
function createMainWindow() {
    mainWindow = new BrowserWindow({
        width: 1440,
        height: 920,
        minWidth: 1080,
        minHeight: 700,
        backgroundColor: '#050505',
        title: 'ExamGuard - AI Exam Environment Monitoring System',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true
        }
    });

    // Remove standard top menu bar for clean modern look
    mainWindow.setMenuBarVisibility(false);

    // Load the dashboard
    mainWindow.loadURL(BACKEND_URL);

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

/**
 * Checks for GitHub updates and shows interactive prompt dialogs.
 */
function setupAutoUpdater() {
    // Only check updates in packaged app to avoid dev errors
    if (!app.isPackaged) {
        console.log('[AutoUpdater] Development mode: skipping GitHub release check.');
        return;
    }

    autoUpdater.on('checking-for-update', () => {
        console.log('[AutoUpdater] Checking for updates on GitHub...');
    });

    autoUpdater.on('update-available', (info) => {
        console.log('[AutoUpdater] Update available:', info.version);
        dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: 'ExamGuard Update Available',
            message: `A new version of ExamGuard (v${info.version}) is available.`,
            detail: `Would you like to download and install this update now?\n\nRelease Notes:\n${info.releaseNotes || 'General improvements and detection updates.'}`,
            buttons: ['Download Update', 'Remind Me Later'],
            defaultId: 0,
            cancelId: 1
        }).then((result) => {
            if (result.response === 0) {
                autoUpdater.downloadUpdate();
                if (Notification.isSupported()) {
                    new Notification({
                        title: 'ExamGuard Update',
                        body: 'Downloading update in the background...'
                    }).show();
                }
            }
        });
    });

    autoUpdater.on('update-not-available', () => {
        console.log('[AutoUpdater] App is up to date.');
    });

    autoUpdater.on('download-progress', (progressObj) => {
        console.log(`[AutoUpdater] Download speed: ${progressObj.bytesPerSecond} - Downloaded ${progressObj.percent.toFixed(1)}%`);
    });

    autoUpdater.on('update-downloaded', (info) => {
        console.log('[AutoUpdater] Update downloaded.');
        dialog.showMessageBox(mainWindow, {
            type: 'question',
            title: 'Update Ready to Install',
            message: `ExamGuard v${info.version} has been downloaded.`,
            detail: 'Restart ExamGuard now to apply the update?',
            buttons: ['Restart Now', 'Later'],
            defaultId: 0,
            cancelId: 1
        }).then((result) => {
            if (result.response === 0) {
                autoUpdater.quitAndInstall();
            }
        });
    });

    autoUpdater.on('error', (err) => {
        console.warn('[AutoUpdater] Update check error:', err.message);
    });

    // Check on startup
    autoUpdater.checkForUpdates();
}

/**
 * Gracefully shuts down the Python backend child process.
 */
function cleanupPythonBackend() {
    if (pythonProcess) {
        console.log('[ExamGuard Desktop] Terminating Python backend...');
        try {
            if (process.platform === 'win32') {
                spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
            } else {
                pythonProcess.kill('SIGINT');
                // Force kill if not dead after 2s
                setTimeout(() => {
                    if (pythonProcess) pythonProcess.kill('SIGKILL');
                }, 2000);
            }
        } catch (e) {
            console.error('[ExamGuard Desktop] Error killing Python process:', e);
        }
        pythonProcess = null;
    }
}

// App Lifecycle
app.whenReady().then(() => {
    startPythonBackend();

    waitForBackend(() => {
        createMainWindow();
        setupAutoUpdater();
    });

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createMainWindow();
        }
    });
});

app.on('window-all-closed', () => {
    cleanupPythonBackend();
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', () => {
    cleanupPythonBackend();
});

