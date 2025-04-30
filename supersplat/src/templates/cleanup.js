// SuperSplat Viewer Cleanup Script
// This script automatically cleans up the viewer interface for end users

// Wait for page to fully load
window.addEventListener('load', function() {
    console.log('SuperSplat Viewer Cleanup Script running...');
    
    // Function to remove development buttons
    function removeDevButtons() {
        const buttonTexts = [
            'Hide Mesh Objects',
            'Show Mesh Objects', 
            'Rotate Left',
            'Rotate Right',
            'Scale +',
            'Scale -',
            'Fix Cameras',
            'Load Camera Data',
            'Export Viewer'
        ];
        
        // Keep checking for and removing buttons
        function checkAndRemoveButtons() {
            let found = false;
            const allButtons = document.querySelectorAll('button');
            allButtons.forEach(button => {
                const text = button.textContent.trim();
                if (buttonTexts.includes(text)) {
                    if (button.parentNode) {
                        button.parentNode.removeChild(button);
                        found = true;
                        console.log('Removed button:', text);
                    }
                }
            });
            return found;
        }
        
        // Check immediately
        checkAndRemoveButtons();
        
        // And check periodically for 10 seconds
        let checkCount = 0;
        const buttonCheckInterval = setInterval(() => {
            const found = checkAndRemoveButtons();
            checkCount++;
            if (checkCount >= 20 || !found) {
                clearInterval(buttonCheckInterval);
            }
        }, 500);
    }
    
    // Function to enable camera detection
    function enableCameraDetection() {
        function findAndClickButton() {
            const cameraButton = Array.from(document.querySelectorAll('button')).find(
                button => button.textContent === 'Enable Camera Detection'
            );
            
            if (cameraButton) {
                cameraButton.style.top = '10px';
                cameraButton.click();
                console.log('Camera detection enabled');
                return true;
            }
            return false;
        }
        
        // Try immediately
        findAndClickButton();
        
        // Try again after a delay
        setTimeout(findAndClickButton, 1000);
        
        // And keep checking
        let checkCount = 0;
        const cameraCheckInterval = setInterval(() => {
            const found = findAndClickButton();
            checkCount++;
            if (checkCount >= 10 || found) {
                clearInterval(cameraCheckInterval);
            }
        }, 1000);
    }
    
    // Function to hide debug objects
    function hideDebugObjects() {
        function tryHideObjects() {
            if (!window.meshObjects) return false;
            
            // Hide debug objects
            if (window.meshObjects.debugObjects) {
                window.meshObjects.debugObjects.forEach(obj => {
                    if (obj && obj.enabled !== undefined) {
                        obj.enabled = false;
                    }
                });
            }
            
            // Hide camera markers
            if (window.meshObjects.cameraMarkers) {
                window.meshObjects.cameraMarkers.forEach(marker => {
                    if (marker && marker.enabled !== undefined) {
                        marker.enabled = false;
                    }
                });
            }
            
            // Force render update
            if (window.app && window.app.renderNextFrame) {
                window.app.renderNextFrame = true;
            }
            
            console.log('Debug objects hidden');
            return true;
        }
        
        // Try immediately
        tryHideObjects();
        
        // Keep trying until successful
        let checkCount = 0;
        const objCheckInterval = setInterval(() => {
            const success = tryHideObjects();
            checkCount++;
            if (checkCount >= 20 || success) {
                clearInterval(objCheckInterval);
            }
        }, 500);
    }
    
    // Function to fix canvas visibility
    function fixCanvasVisibility() {
        function tryFixCanvas() {
            // Find all canvases
            const canvases = document.querySelectorAll('canvas');
            
            // Make all canvases visible
            if (canvases.length > 0) {
                canvases.forEach((canvas, index) => {
                    // Set z-index to ensure visibility
                    canvas.style.zIndex = '5';
                    canvas.style.position = 'relative';
                    canvas.style.visibility = 'visible';
                    canvas.style.opacity = '1';
                    
                    // If it has an absolute parent, adjust that too
                    let parent = canvas.parentElement;
                    while (parent && parent !== document.body) {
                        const style = window.getComputedStyle(parent);
                        if (style.position === 'absolute' || style.position === 'fixed') {
                            parent.style.zIndex = '5';
                            parent.style.visibility = 'visible';
                            parent.style.opacity = '1';
                        }
                        parent = parent.parentElement;
                    }
                });
                console.log('Canvas visibility fixed');
                return true;
            }
            return false;
        }
        
        // Try immediately
        tryFixCanvas();
        
        // And try again after a delay
        setTimeout(tryFixCanvas, 1000);
        
        // Keep checking for a while
        let checkCount = 0;
        const canvasCheckInterval = setInterval(() => {
            const found = tryFixCanvas();
            checkCount++;
            if (checkCount >= 10) {
                clearInterval(canvasCheckInterval);
            }
        }, 1000);
    }
    
    // Function to add watermark
    function addWatermark() {
        if (!document.getElementById('supersplat-watermark')) {
            const watermark = document.createElement('div');
            watermark.id = 'supersplat-watermark';
            watermark.textContent = 'SuperSplat Viewer';
            watermark.style.position = 'absolute';
            watermark.style.bottom = '10px';
            watermark.style.right = '10px';
            watermark.style.color = 'rgba(255,255,255,0.5)';
            watermark.style.fontSize = '12px';
            watermark.style.pointerEvents = 'none';
            watermark.style.zIndex = '1000';
            document.body.appendChild(watermark);
            console.log('Watermark added');
        }
    }
    
    // Add cleanup CSS
    function addCleanupStyles() {
        const styleElem = document.createElement('style');
        styleElem.textContent = `
            /* Ensure canvases are visible */
            canvas {
                z-index: 5 !important;
                visibility: visible !important;
                opacity: 1 !important;
                position: relative !important;
            }
            
            /* Hide any popup interfaces */
            div[style*="position: fixed"][style*="z-index: 10000"],
            div[id="cameraPopup"] {
                display: none !important;
            }
        `;
        document.head.appendChild(styleElem);
        console.log('Cleanup styles added');
    }
    
    // Run all cleanup functions
    removeDevButtons();
    enableCameraDetection();
    hideDebugObjects();
    fixCanvasVisibility();
    addWatermark();
    addCleanupStyles();
    
    // Periodically check for canvas issues and fix them
    setInterval(fixCanvasVisibility, 2000);
    
    console.log('SuperSplat Viewer Cleanup Script completed');
});