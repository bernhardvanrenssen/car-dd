import { BoundingBox, Color, Mat4, Script, Vec3, Entity, StandardMaterial, Quat, BLEND_NONE, BLEND_NORMAL, LAYER_WORLD, Ray } from 'playcanvas';

import { CubicSpline } from 'spline';

function logWithTimestamp(message) {
    const now = new Date();
    const timestamp = `${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}.${now.getMilliseconds()}`;
    console.log(`[${timestamp}] ${message}`);
}

// ——— AABB ↔️ Ray intersection helper ———
function intersectAABB(ray, aabb) {
    // Inverse of ray direction
    const inv = new Vec3(1 / ray.direction.x, 1 / ray.direction.y, 1 / ray.direction.z);
    // Min/max corners
    const mn = aabb.center.clone().sub(aabb.halfExtents);
    const mx = aabb.center.clone().add(aabb.halfExtents);

    let tmin = (mn.x - ray.origin.x) * inv.x;
    let tmax = (mx.x - ray.origin.x) * inv.x;
    if (inv.x < 0) [tmin, tmax] = [tmax, tmin];

    let tymin = (mn.y - ray.origin.y) * inv.y;
    let tymax = (mx.y - ray.origin.y) * inv.y;
    if (inv.y < 0) [tymin, tymax] = [tymax, tymin];

    if (tmin > tymax || tymin > tmax) return null;
    tmin = Math.max(tmin, tymin);
    tmax = Math.min(tmax, tymax);

    let tzmin = (mn.z - ray.origin.z) * inv.z;
    let tzmax = (mx.z - ray.origin.z) * inv.z;
    if (inv.z < 0) [tzmin, tzmax] = [tzmax, tzmin];

    if (tmin > tzmax || tzmin > tmax) return null;
    tmin = Math.max(tmin, tzmin);

    // Only care about the first entry (front face)
    return tmin >= 0 ? tmin : null;
}

const nearlyEquals = (a, b, epsilon = 1e-4) => {
    return !a.some((v, i) => Math.abs(v - b[i]) >= epsilon);
};

const url = new URL(location.href);

const params = {
    noui: url.searchParams.has('noui'),
    noanim: url.searchParams.has('noanim'),
    posterUrl: url.searchParams.get('poster')
};

// display a blurry poster image which resolves to sharp during loading
class Poster {
    constructor(url) {
        const blur = (progress) => `blur(${Math.floor((100 - progress) * 0.4)}px)`;

        const element = document.getElementById('poster');
        element.style.backgroundImage = `url(${url})`;
        element.style.display = 'block';
        element.style.filter = blur(0);

        this.progress = (progress) => {
            element.style.filter = blur(progress);
        };

        this.hide = () => {
            element.style.display = 'none';
        };
    }
}

const poster = params.posterUrl && new Poster(params.posterUrl);

class FrameScene extends Script {
    initialize() {
        const { settings } = this;
        const { camera, animTracks } = settings;
        const { position, target } = camera;

        this.position = position && new Vec3(position);
        this.target = target && new Vec3(target);

        // construct camera animation track
        if (animTracks?.length > 0 && settings.camera.startAnim === 'animTrack') {
            const track = animTracks.find(track => track.name === camera.animTrack);
            if (track) {
                const { keyframes, duration } = track;
                const { times, values } = keyframes;
                const { position, target } = values;

                // construct the points array containing position and target
                const points = [];
                for (let i = 0; i < times.length; i++) {
                    points.push(position[i * 3], position[i * 3 + 1], position[i * 3 + 2]);
                    points.push(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
                }

                this.cameraAnim = {
                    time: 0,
                    spline: CubicSpline.fromPointsLooping(duration, times, points),
                    track,
                    result: []
                };
            }
        }
    }

    frameScene(bbox, smooth = true) {
        const sceneSize = bbox.halfExtents.length();
        const distance = sceneSize / Math.sin(this.entity.camera.fov / 180 * Math.PI * 0.5);
        this.entity.script.cameraControls.sceneSize = sceneSize;
        this.entity.script.cameraControls.focus(bbox.center, new Vec3(2, 1, 2).normalize().mulScalar(distance).add(bbox.center), smooth);
    }

    resetCamera(bbox, smooth = true) {
        const sceneSize = bbox.halfExtents.length();
        this.entity.script.cameraControls.sceneSize = sceneSize * 0.2;
        this.entity.script.cameraControls.focus(this.target ?? Vec3.ZERO, this.position ?? new Vec3(2, 1, 2), smooth);
    }

    initCamera() {
        const { app } = this;
        const { graphicsDevice } = app;
        let animating = false;
        let animationTimer = 0;

        // get the gsplat component
        const gsplatComponent = app.root.findComponent('gsplat');

        // calculate the bounding box
        const bbox = gsplatComponent?.instance?.meshInstance?.aabb ?? new BoundingBox();
        if (bbox.halfExtents.length() > 100 || this.position || this.target) {
            this.resetCamera(bbox, false);
        } else {
            this.frameScene(bbox, false);
        }

        const cancelAnimation = () => {
            if (animating) {
                animating = false;

                // copy current camera position and target
                const r = this.cameraAnim.result;
                this.entity.script.cameraControls.focus(
                    new Vec3(r[3], r[4], r[5]),
                    new Vec3(r[0], r[1], r[2]),
                    false
                );
            }
        };

        // listen for interaction events
        const events = [ 'wheel', 'pointerdown', 'contextmenu' ];
        const handler = (e) => {
            cancelAnimation();
            events.forEach(event => app.graphicsDevice.canvas.removeEventListener(event, handler));
        };
        events.forEach(event => app.graphicsDevice.canvas.addEventListener(event, handler));

        window.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.altKey || e.metaKey) return;

            switch (e.key) {
                case 'f':
                    cancelAnimation();
                    this.frameScene(bbox);
                    break;
                case 'r':
                    cancelAnimation();
                    this.resetCamera(bbox);
                    break;
            }
        });

        app.on('update', (deltaTime) => {
            // handle camera animation
            if (this.cameraAnim && animating && !params.noanim) {
                const { cameraAnim } = this;
                const { spline, track, result } = cameraAnim;

                // update animation timer
                animationTimer += deltaTime;

                // update the track cursor
                if (animationTimer < 5) {
                    // ease in
                    cameraAnim.time += deltaTime * Math.pow(animationTimer / 5, 0.5);
                } else {
                    cameraAnim.time += deltaTime;
                }

                if (cameraAnim.time >= track.duration) {
                    switch (track.loopMode) {
                        case 'none': cameraAnim.time = track.duration; break;
                        case 'repeat': cameraAnim.time = cameraAnim.time % track.duration; break;
                        case 'pingpong': cameraAnim.time = cameraAnim.time % (track.duration * 2); break;
                    }
                }

                // evaluate the spline
                spline.evaluate(cameraAnim.time > track.duration ? track.duration - cameraAnim.time : cameraAnim.time, result);

                // set camera
                this.entity.setPosition(result[0], result[1], result[2]);
                this.entity.lookAt(result[3], result[4], result[5]);
            }
        });

        const prevProj = new Mat4();
        const prevWorld = new Mat4();

        app.on('framerender', () => {
            if (!app.autoRender && !app.renderNextFrame) {
                const world = this.entity.getWorldTransform();
                if (!nearlyEquals(world.data, prevWorld.data)) {
                    app.renderNextFrame = true;
                }

                const proj = this.entity.camera.projectionMatrix;
                if (!nearlyEquals(proj.data, prevProj.data)) {
                    app.renderNextFrame = true;
                }

                if (app.renderNextFrame) {
                    prevWorld.copy(world);
                    prevProj.copy(proj);
                }
            }
        });

        // wait for first gsplat sort
        const handle = gsplatComponent?.instance?.sorter?.on('updated', () => {
            handle.off();

            // request frame render
            app.renderNextFrame = true;

            // wait for first render to complete
            const frameHandle = app.on('frameend', () => {
                frameHandle.off();

                // hide loading indicator
                document.getElementById('loadingWrap').classList.add('hidden');

                // fade out poster
                poster?.hide();

                // start animating once the first frame is rendered
                if (this.cameraAnim) {
                    animating = true;
                }

                // emit first frame event on window
                window.firstFrame?.();
            });
        });

        const updateHorizontalFov = (width, height) => {
            this.entity.camera.horizontalFov = width > height;
        };

        // handle fov on canvas resize
        graphicsDevice.on('resizecanvas', (width, height) => {
            updateHorizontalFov(width, height);
            app.renderNextFrame = true;
        });

        // configure on-demand rendering
        app.autoRender = false;
        updateHorizontalFov(graphicsDevice.width, graphicsDevice.height);
    }

    postInitialize() {
        const assets = this.app.assets.filter(asset => asset.type === 'gsplat');
        if (assets.length > 0) {
            const asset = assets[0];
            if (asset.loaded) {
                this.initCamera();
            } else {
                asset.on('load', () => {
                    this.initCamera();
                });
            }
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    const appElement = await document.querySelector('pc-app').ready();
    const cameraElement = await document.querySelector('pc-entity[name="camera"]').ready();

    const app = await appElement.app;
    const camera = cameraElement.entity;
    const settings = await window.settings;

    // Store reference to camera for later use with click detection
    window.appCamera = camera;

    camera.camera.clearColor = new Color(settings.background.color);
    camera.camera.fov = settings.camera.fov;
    camera.script.create(FrameScene, {
        properties: { settings }
    });

    // Initialize structure to store camera markers
    window.cameraData = [];

    // Auto-load camera positions from camera_positions.json
    loadCameraPositions();

    // Add click-to-view functionality
    setupClickToView(app, camera);

    // Update loading indicator
    const assets = app.assets.filter(asset => asset.type === 'gsplat');
    if (assets.length > 0) {
        const asset = assets[0];
        const loadingText = document.getElementById('loadingText');
        const loadingBar = document.getElementById('loadingBar');
        asset.on('progress', (received, length) => {
            const v = (Math.min(1, received / length) * 100).toFixed(0);
            loadingText.textContent = `${v}%`;
            loadingBar.style.backgroundImage = 'linear-gradient(90deg, #F60 0%, #F60 ' + v + '%, white ' + v + '%, white 100%)';
            poster?.progress(v);
        });
    }

    // On entering/exiting AR, we need to set the camera clear color to transparent black
    let cameraEntity, skyType = null;
    const clearColor = new Color();

    app.xr.on('start', () => {
        if (app.xr.type === 'immersive-ar') {
            cameraEntity = app.xr.camera;
            clearColor.copy(cameraEntity.camera.clearColor);
            cameraEntity.camera.clearColor = new Color(0, 0, 0, 0);

            const sky = document.querySelector('pc-sky');
            if (sky && sky.type !== 'none') {
                skyType = sky.type;
                sky.type = 'none';
            }

            app.autoRender = true;
        }
    });

    app.xr.on('end', () => {
        if (app.xr.type === 'immersive-ar') {
            cameraEntity.camera.clearColor = clearColor;

            const sky = document.querySelector('pc-sky');
            if (sky) {
                if (skyType) {
                    sky.type = skyType;
                    skyType = null;
                } else {
                    sky.removeAttribute('type');
                }
            }

            app.autoRender = false;
        }
    });

    // Get button and info panel elements
    const dom = ['arMode', 'vrMode', 'enterFullscreen', 'exitFullscreen', 'info', 'infoPanel', 'buttonContainer'].reduce((acc, id) => {
        acc[id] = document.getElementById(id);
        return acc;
    }, {});

    // AR
    if (app.xr.isAvailable('immersive-ar')) {
        dom.arMode.classList.remove('hidden');
        dom.arMode.addEventListener('click', () => app.xr.start(app.root.findComponent('camera'), 'immersive-ar', 'local-floor'));
    }

    // VR
    if (app.xr.isAvailable('immersive-vr')) {
        dom.vrMode.classList.remove('hidden');
        dom.vrMode.addEventListener('click', () => app.xr.start(app.root.findComponent('camera'), 'immersive-vr', 'local-floor'));
    }

    // Fullscreen
    if (document.documentElement.requestFullscreen && document.exitFullscreen) {
        dom.enterFullscreen.classList.remove('hidden');
        dom.enterFullscreen.addEventListener('click', () => document.documentElement.requestFullscreen());
        dom.exitFullscreen.addEventListener('click', () => document.exitFullscreen());
        document.addEventListener('fullscreenchange', () => {
            dom.enterFullscreen.classList[document.fullscreenElement ? 'add' : 'remove']('hidden');
            dom.exitFullscreen.classList[document.fullscreenElement ? 'remove' : 'add']('hidden');
        });
    }

    // Info
    dom.info.addEventListener('click', () => {
        dom.infoPanel.classList.toggle('hidden');
    });

    // Keyboard handler
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            if (app.xr.active) {
                app.xr.end();
            }
            dom.infoPanel.classList.add('hidden');
            
            // Also close any open image viewer
            const overlay = document.getElementById('fullscreenOverlay');
            if (overlay) {
                document.body.removeChild(overlay);
            }
        }
    });

    // Hide UI
    if (params.noui) {
        dom.buttonContainer.classList.add('hidden');
    }
});

// Function to load camera positions from JSON file
async function loadCameraPositions() {
    try {
        const response = await fetch('camera_positions.json');
        if (!response.ok) {
            console.warn('Camera positions file not found, click-to-view functionality will be disabled');
            return;
        }
        
        const cameraPositions = await response.json();
        console.log(`Loaded ${cameraPositions.length} camera positions`);
        
        // Store camera positions in global variable for access by click handler
        window.cameraData = cameraPositions;
        
        // Create a "View HD Images" button to inform users about the functionality
        createViewHDButton();
    } catch (err) {
        console.error('Error loading camera positions:', err);
    }
}

// Function to create a button informing users about click-to-view functionality
function createViewHDButton() {
    if (!window.cameraData || window.cameraData.length === 0) return;
    
    const button = document.createElement('button');
    button.textContent = 'Click Model to View HD Images';
    button.style.position = 'absolute';
    button.style.zIndex = '1000';
    button.style.padding = '8px 16px';
    button.style.backgroundColor = '#007bff';
    button.style.color = 'white';
    button.style.border = 'none';
    button.style.borderRadius = '4px';
    button.style.cursor = 'pointer';
    button.style.top = '10px';
    button.style.left = '50%';
    button.style.transform = 'translateX(-50%)';
    button.style.opacity = '0.8';
    
    // Fade out button after 5 seconds
    setTimeout(() => {
        button.style.transition = 'opacity 1s ease-out';
        button.style.opacity = '0';
        // Remove from DOM after fade out
        setTimeout(() => {
            if (button.parentNode) {
                button.parentNode.removeChild(button);
            }
        }, 1000);
    }, 5000);
    
    document.body.appendChild(button);
}

// Function to setup click-to-view functionality
function setupClickToView(app, camera) {
    logWithTimestamp("Setting up click-to-view functionality");
    
    // Check if camera data is loaded
    if (!window.cameraData) {
        console.error("No cameraData available in global scope");
    } else {
        logWithTimestamp(`Found ${window.cameraData.length} camera positions`);
    }
    
    // Create debug sphere for indicating click point (invisible initially)
    window.debugSphere = new Entity('debugSphere');
    window.debugSphere.addComponent('model', { type: 'sphere' });
    window.debugSphere.setLocalScale(0.1, 0.1, 0.1);
    
    const mat = new StandardMaterial();
    mat.diffuse = new Color(0.5, 0, 0.5);
    mat.emissive = new Color(0.5, 0, 0.5);
    mat.emissiveIntensity = 0.8;
    mat.depthTest = false;
    mat.depthWrite = false;
    mat.update();
    
    window.debugSphere.model.material = mat;
    window.debugSphere.model.meshInstances.forEach(mi => {
        mi.layer = LAYER_WORLD + 1;
        mi.drawOrder = 9999;
    });
    
    window.debugSphere.enabled = false; // Hide initially
    app.root.addChild(window.debugSphere);
    logWithTimestamp("Debug sphere created");
    
    // Setup click handler
    const canvas = document.querySelector('canvas');
    if (!canvas) {
        console.error('Canvas element not found, click-to-view functionality will be disabled');
        return;
    }
    
    logWithTimestamp("Adding click listener to canvas");
    canvas.addEventListener('click', (event) => {
        logWithTimestamp("Canvas clicked");
        
        if (!window.cameraData || window.cameraData.length === 0) {
            console.error("No camera data available for click detection");
            return;
        }
        
        // Mouse coords relative to canvas
        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        logWithTimestamp(`Click at canvas coordinates: (${mouseX}, ${mouseY})`);
        
        // Build pick ray
        const nearPoint = new Vec3();
        const farPoint = new Vec3();
        camera.camera.screenToWorld(mouseX, mouseY, camera.camera.nearClip, nearPoint);
        camera.camera.screenToWorld(mouseX, mouseY, camera.camera.farClip, farPoint);
        const direction = farPoint.sub(nearPoint).normalize();
        const ray = new Ray(nearPoint, direction);
        
        logWithTimestamp(`Ray origin: ${nearPoint.toString()}, direction: ${direction.toString()}`);
        
        // Intersect with AABB
        const gs = app.root.findComponent('gsplat');
        if (!gs) {
            console.error("No gsplat component found");
            return;
        }
        
        if (!gs.instance) {
            console.error("gsplat instance not found");
            return;
        }
        
        if (!gs.instance.meshInstance) {
            console.error("gsplat meshInstance not found");
            return;
        }
        
        if (!gs.instance.meshInstance.aabb) {
            console.error("gsplat aabb not found");
            return;
        }
        
        logWithTimestamp(`AABB center: ${gs.instance.meshInstance.aabb.center.toString()}, half extents: ${gs.instance.meshInstance.aabb.halfExtents.toString()}`);
        
        const tNear = intersectAABB(ray, gs.instance.meshInstance.aabb);
        logWithTimestamp(`Ray intersection result: ${tNear}`);
        
        if (tNear === null) {
            logWithTimestamp("No intersection with model, removing any existing popup");
            // Remove any existing popup when clicking empty space
            const popup = document.getElementById('cameraPopup');
            if (popup) popup.remove();
            
            return;
        }
        
        // Compute click point
        const clickPoint = direction.mulScalar(tNear).add(nearPoint);
        logWithTimestamp(`Click point in 3D space: ${clickPoint.toString()}`);
        
        // Show debug sphere at click point
        window.debugSphere.setPosition(clickPoint.x, clickPoint.y, clickPoint.z);
        window.debugSphere.enabled = true;
        app.renderNextFrame = true;
        logWithTimestamp("Debug sphere positioned at click point and enabled");
        
        // Find closest camera by weighted distance
        const horizontalWeight = 0.05;
        let minDist = Infinity;
        let bestCamera = null;
        
        window.cameraData.forEach((cameraInfo, index) => {
            if (!cameraInfo.position) {
                console.error(`Camera at index ${index} is missing position data:`, cameraInfo);
                return;
            }
            
            const pos = cameraInfo.position;
            const dx = pos.x - clickPoint.x;
            const dy = pos.y - clickPoint.y;
            const dz = pos.z - clickPoint.z;
            
            const dist = Math.sqrt(
                horizontalWeight * (dx*dx + dz*dz) + (dy*dy)
            );
            
            if (dist < minDist) {
                minDist = dist;
                bestCamera = cameraInfo;
            }
        });
        
        logWithTimestamp(`Best camera search result - distance: ${minDist}, found: ${!!bestCamera}`);
        
        if (bestCamera) {
            logWithTimestamp(`Selected camera: ${JSON.stringify(bestCamera, null, 2)}`);
            
            // Remove any existing popup
            const oldPopup = document.getElementById('cameraPopup');
            if (oldPopup) {
                logWithTimestamp("Removing existing popup");
                oldPopup.remove();
            }
            
            // Create popup with thumbnail and expand button
            const container = document.createElement('div');
            container.id = 'cameraPopup';
            container.style.position = 'absolute';
            container.style.left = `${event.clientX}px`;
            container.style.top = `${event.clientY}px`;
            container.style.display = 'flex';
            container.style.flexDirection = 'column';
            container.style.border = '1px solid rgba(0,0,0,0.5)';
            container.style.boxShadow = '0 0 6px rgba(0,0,0,0.3)';
            container.style.background = 'white';
            container.style.zIndex = '9000';
            container.style.overflow = 'hidden'; // Prevent content from overflowing
            container.style.padding = '0'; // Remove any padding
            container.style.maxWidth = '100px'; // Match image width exactly

            // Add thumbnail
            const img = document.createElement('img');
            img.src = `images/${bestCamera.imageName}`;
            img.style.width = '100%'; // Fill the container width completely
            img.style.height = 'auto';
            img.style.display = 'block'; // Remove any inline spacing issues
            img.style.pointerEvents = 'none';
            img.style.margin = '0'; // Remove any margins
            container.appendChild(img);

            // Add expand button with better touch target but thinner height
            const expandBtn = document.createElement('button');
            expandBtn.textContent = 'View Full Size';
            expandBtn.style.padding = '6px 0'; // Reduced vertical padding
            expandBtn.style.margin = '0'; // Remove margins
            expandBtn.style.cursor = 'pointer';
            expandBtn.style.background = '#007bff';
            expandBtn.style.color = 'white';
            expandBtn.style.border = 'none';
            expandBtn.style.fontSize = '12px'; // Slightly smaller font
            expandBtn.style.width = '100%'; // Full width of container
            expandBtn.style.minHeight = '32px'; // Reduced height, still reasonable for touch

            // Handle expand button click and touch
            const openFullScreenViewer = (e) => {
                e.stopPropagation();
                openFullsizeViewer(bestCamera);
            };

            expandBtn.addEventListener('click', openFullScreenViewer);
            expandBtn.addEventListener('touchend', (e) => {
                e.preventDefault(); // Prevent ghost clicks
                openFullScreenViewer(e);
            });

            container.appendChild(expandBtn);
            document.body.appendChild(container);

            // THE KEY FIX: Use setTimeout to add the document click listener on the next tick
            // This prevents the current click from immediately closing the popup
            setTimeout(() => {
                // Make the popup automatically close when clicking outside of it
                document.addEventListener('click', function closePopup(e) {
                    if (!container.contains(e.target) && e.target !== container) {
                        if (container.parentNode) {
                            container.parentNode.removeChild(container);
                        }
                        document.removeEventListener('click', closePopup);
                    }
                });
            }, 0);

            // Close popup after 10 seconds to prevent lingering popups
            setTimeout(() => {
                if (container.parentNode) {
                    container.parentNode.removeChild(container);
                }
            }, 10000);
        } else {
            console.error("Could not find a camera close to click point");
        }
    });
    
    logWithTimestamp("Click-to-view setup complete");
}

// Add this diagnostic function to check the camera_positions.json file
async function checkCameraPositionsFile() {
    try {
        const response = await fetch('camera_positions.json');
        if (!response.ok) {
            console.error(`Error fetching camera_positions.json: ${response.status} ${response.statusText}`);
            return;
        }
        
        const data = await response.json();
        console.log(`Successfully loaded camera_positions.json with ${data.length} entries`);
        
        // Check a sample entry
        if (data.length > 0) {
            console.log("Sample camera entry:", data[0]);
            console.log("Position data type check:", 
                typeof data[0].position === 'object',
                data[0].position !== null,
                data[0].position.hasOwnProperty('x'),
                data[0].position.hasOwnProperty('y'),
                data[0].position.hasOwnProperty('z')
            );
        }
        
        // Check if image files exist
        if (data.length > 0 && data[0].imageName) {
            const testImg = new Image();
            testImg.onload = () => console.log(`Test image found: images/${data[0].imageName}`);
            testImg.onerror = () => console.error(`Test image NOT found: images/${data[0].imageName}`);
            testImg.src = `images/${data[0].imageName}`;
        }
        
    } catch (err) {
        console.error("Error checking camera positions file:", err);
    }
}

// Function to find nearest camera in a direction
function findNearestCameraInDirection(currentCamera, isRightDirection) {
    if (!window.cameraData || window.cameraData.length <= 1) return null;
    
    const currentPos = currentCamera.position;
    
    // Calculate center of all cameras
    const center = calculateCamerasCenter();
    
    // Vector from center to current camera
    const markerVec = new Vec3(
        currentPos.x - center.x,
        0, // Ignore Y for horizontal navigation
        currentPos.z - center.z
    ).normalize();
    
    // Compute right vector (perpendicular to marker vector)
    const worldUp = new Vec3(0, 1, 0);
    const rightVec = new Vec3().cross(worldUp, markerVec).normalize();
    
    // For each camera, compute how far it is in right or left direction
    const camerasWithDirectionality = window.cameraData
        .filter(camera => camera !== currentCamera)
        .map(camera => {
            const pos = camera.position;
            
            // Vector from current camera to other camera
            const toMarkerVec = new Vec3(
                pos.x - currentPos.x,
                0, // Ignore Y for horizontal navigation
                pos.z - currentPos.z
            );
            
            // If toMarkerVec is zero length, skip this marker
            if (toMarkerVec.length() < 0.001) return { camera, score: -Infinity };
            
            toMarkerVec.normalize();
            
            // Dot product with right vector gives how far right the camera is
            // Positive is right, negative is left
            const rightAmount = rightVec.dot(toMarkerVec);
            
            // Forward amount
            const forwardAmount = markerVec.dot(toMarkerVec);
            
            // Distance factor
            const distance = Math.sqrt(
                Math.pow(pos.x - currentPos.x, 2) +
                Math.pow(pos.z - currentPos.z, 2)
            );
            
            // Score combines direction and distance
            const directionScore = isRightDirection ? rightAmount : -rightAmount;
            
            // Higher score = better match (more in right direction, more forward, closer)
            const score = directionScore * 3 + forwardAmount * 2 - (distance * 0.1);
            
            return { camera, score };
        });
    
    // Find the camera with highest score
    camerasWithDirectionality.sort((a, b) => b.score - a.score);
    
    // Return best camera or null if none found
    return camerasWithDirectionality.length > 0 && 
        camerasWithDirectionality[0].score > -1 ? 
        camerasWithDirectionality[0].camera : null;
}

// Function to calculate center of all cameras
function calculateCamerasCenter() {
    if (!window.cameraData || window.cameraData.length === 0) {
        return new Vec3(0, 0, 0);
    }
    
    const center = new Vec3(0, 0, 0);
    let count = 0;
    
    window.cameraData.forEach(camera => {
        if (camera.position) {
            center.x += camera.position.x;
            center.y += camera.position.y;
            center.z += camera.position.z;
            count++;
        }
    });
    
    if (count > 0) {
        center.x /= count;
        center.y /= count;
        center.z /= count;
    }
    
    return center;
}

// Simpler, more direct pinch-to-zoom implementation
function openFullsizeViewer(initialCamera) {
    // Create fullscreen overlay
    const overlay = document.createElement('div');
    overlay.id = 'fullscreenOverlay';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0,0,0,0.9)';
    overlay.style.zIndex = '10000';
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    
    // Create close button
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.style.position = 'absolute';
    closeBtn.style.top = '20px';
    closeBtn.style.right = '20px';
    closeBtn.style.padding = '10px 20px';
    closeBtn.style.background = '#007bff';
    closeBtn.style.color = 'white';
    closeBtn.style.border = 'none';
    closeBtn.style.borderRadius = '4px';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.fontSize = '16px';
    closeBtn.addEventListener('click', () => {
        document.body.removeChild(overlay);
    });
    
    // Current camera reference for navigation
    let currentCamera = initialCamera;
    
    // Create image container - simpler structure to avoid positioning complexities
    const imgContainer = document.createElement('div');
    imgContainer.style.position = 'relative';
    imgContainer.style.width = '80%';
    imgContainer.style.height = '80%';
    imgContainer.style.display = 'flex';
    imgContainer.style.justifyContent = 'center';
    imgContainer.style.alignItems = 'center';
    
    // Create fullsize image with direct transforms
    const fullImg = document.createElement('img');
    fullImg.src = `images/${currentCamera.imageName}`;
    fullImg.style.maxWidth = '100%';
    fullImg.style.maxHeight = '100%';
    fullImg.style.objectFit = 'contain';
    fullImg.style.touchAction = 'none'; // Prevent default touch actions
    
    // Zoom and pan variables 
    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let lastX = 0;
    let lastY = 0;
    let lastDistance = 0;
    
    // Function to update image when changing
    const updateFullImage = () => {
        fullImg.src = `images/${currentCamera.imageName}`;
        // Reset zoom and position
        scale = 1;
        translateX = 0;
        translateY = 0;
        updateTransform();
    };
    
    // Function to update transform
    const updateTransform = () => {
        fullImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    };
    
    // --------- TOUCH EVENTS ---------
    
    // Touch move handler with direct pinch-to-zoom implementation
    imgContainer.addEventListener('touchmove', function(e) {
        e.preventDefault();
        
        // Get touch points
        const touches = e.touches;
        
        // Handle pinch-to-zoom (2 fingers)
        if (touches.length === 2) {
            // Calculate current distance between touch points
            const touch1 = touches[0];
            const touch2 = touches[1];
            const currentDistance = Math.sqrt(
                Math.pow(touch2.clientX - touch1.clientX, 2) + 
                Math.pow(touch2.clientY - touch1.clientY, 2)
            );
            
            // Calculate midpoint (center) of the two touches
            const midpointX = (touch1.clientX + touch2.clientX) / 2;
            const midpointY = (touch1.clientY + touch2.clientY) / 2;
            
            // Get image position relative to viewport
            const rect = fullImg.getBoundingClientRect();
            const imgCenterX = rect.left + rect.width / 2;
            const imgCenterY = rect.top + rect.height / 2;
            
            // If first move in this gesture, initialize lastDistance
            if (lastDistance === 0) {
                lastDistance = currentDistance;
                lastX = midpointX;
                lastY = midpointY;
                return;
            }
            
            // Calculate scale change based on finger distance change
            const scaleChange = currentDistance / lastDistance;
            const newScale = Math.max(1, Math.min(5, scale * scaleChange));
            
            if (newScale !== scale) {
                // Calculate translation to keep the pinch center fixed
                const dx = midpointX - lastX;
                const dy = midpointY - lastY;
                
                // Update scale and translation
                scale = newScale;
                translateX += dx;
                translateY += dy;
                
                // Apply transform
                updateTransform();
                
                // Update last positions
                lastX = midpointX;
                lastY = midpointY;
                lastDistance = currentDistance;
            }
        }
        // Handle pan (1 finger)
        else if (touches.length === 1 && scale > 1) {
            const touch = touches[0];
            
            // If first touch in this gesture, initialize lastX/Y
            if (lastX === 0 && lastY === 0) {
                lastX = touch.clientX;
                lastY = touch.clientY;
                return;
            }
            
            // Calculate change in position
            const dx = touch.clientX - lastX;
            const dy = touch.clientY - lastY;
            
            // Update translation
            translateX += dx;
            translateY += dy;
            
            // Apply transform
            updateTransform();
            
            // Update last position
            lastX = touch.clientX;
            lastY = touch.clientY;
        }
    }, { passive: false });
    
    // Touch start handler
    imgContainer.addEventListener('touchstart', function(e) {
        e.preventDefault();
        
        const touches = e.touches;
        
        if (touches.length === 2) {
            // Initialize for pinch-to-zoom
            const touch1 = touches[0];
            const touch2 = touches[1];
            lastDistance = Math.sqrt(
                Math.pow(touch2.clientX - touch1.clientX, 2) + 
                Math.pow(touch2.clientY - touch1.clientY, 2)
            );
            lastX = (touch1.clientX + touch2.clientX) / 2;
            lastY = (touch1.clientY + touch2.clientY) / 2;
        } 
        else if (touches.length === 1 && scale > 1) {
            // Initialize for pan
            lastX = touches[0].clientX;
            lastY = touches[0].clientY;
        }
    }, { passive: false });
    
    // Touch end handler
    imgContainer.addEventListener('touchend', function(e) {
        // Reset tracking variables when touch ends
        if (e.touches.length === 0) {
            lastDistance = 0;
            lastX = 0;
            lastY = 0;
        }
        // Handle transition from two fingers to one finger
        else if (e.touches.length === 1) {
            lastDistance = 0;
            lastX = e.touches[0].clientX;
            lastY = e.touches[0].clientY;
        }
    });
    
    // Double tap to reset zoom
    let lastTapTime = 0;
    imgContainer.addEventListener('touchend', function(e) {
        const currentTime = new Date().getTime();
        const tapLength = currentTime - lastTapTime;
        
        if (tapLength < 300 && tapLength > 0) {
            // Double tap detected
            scale = 1;
            translateX = 0;
            translateY = 0;
            updateTransform();
            e.preventDefault();
        }
        
        lastTapTime = currentTime;
    });
    
    // --------- MOUSE EVENTS (KEPT FOR DESKTOP) ---------
    
    // Variables for mouse interaction
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    
    // Mouse wheel zoom
    imgContainer.addEventListener('wheel', function(e) {
        e.preventDefault();
        
        // Calculate zoom change
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(1, Math.min(5, scale * delta));
        
        if (newScale !== scale) {
            // Get mouse position
            const mouseX = e.clientX;
            const mouseY = e.clientY;
            
            // Get image position
            const rect = fullImg.getBoundingClientRect();
            
            // Calculate relative mouse position to image center
            const imageCenterX = rect.left + rect.width / 2;
            const imageCenterY = rect.top + rect.height / 2;
            
            // Calculate zoom direction vector from center
            const zoomDirX = mouseX - imageCenterX;
            const zoomDirY = mouseY - imageCenterY;
            
            // Update scale
            scale = newScale;
            
            // Update transform
            updateTransform();
        }
    });
    
    // Mouse down for panning
    imgContainer.addEventListener('mousedown', function(e) {
        // Only allow panning when zoomed in
        if (scale <= 1) return;
        
        isDragging = true;
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
        imgContainer.style.cursor = 'grabbing';
    });
    
    // Mouse move for panning
    document.addEventListener('mousemove', function(e) {
        if (!isDragging) return;
        
        translateX = e.clientX - startX;
        translateY = e.clientY - startY;
        updateTransform();
    });
    
    // Mouse up to end panning
    document.addEventListener('mouseup', function() {
        isDragging = false;
        imgContainer.style.cursor = scale > 1 ? 'grab' : 'default';
    });
    
    // Double click to reset zoom
    imgContainer.addEventListener('dblclick', function() {
        scale = 1;
        translateX = 0;
        translateY = 0;
        updateTransform();
    });
    
    // --------- NAVIGATION ---------
    
    // Add navigation arrows
    const createArrow = (direction) => {
        const arrow = document.createElement('div');
        arrow.style.position = 'absolute';
        arrow.style.top = '50%';
        arrow.style[direction === 'left' ? 'left' : 'right'] = '20px';
        arrow.style.width = '50px';
        arrow.style.height = '50px';
        arrow.style.background = 'rgba(255, 255, 255, 0.3)';
        arrow.style.borderRadius = '50%';
        arrow.style.display = 'flex';
        arrow.style.alignItems = 'center';
        arrow.style.justifyContent = 'center';
        arrow.style.cursor = 'pointer';
        arrow.style.transform = 'translateY(-50%)';
        arrow.style.fontSize = '24px';
        arrow.style.fontWeight = 'bold';
        arrow.style.color = 'white';
        arrow.style.userSelect = 'none';
        arrow.textContent = direction === 'left' ? '←' : '→';
        
        arrow.addEventListener('mouseenter', () => {
            arrow.style.background = 'rgba(255, 255, 255, 0.5)';
        });
        
        arrow.addEventListener('mouseleave', () => {
            arrow.style.background = 'rgba(255, 255, 255, 0.3)';
        });
        
        // Support for both mouse and touch events
        const navigateImage = () => {
            // Find next camera - left means go to camera on the left, right means go to camera on the right
            const nextCamera = findNearestCameraInDirection(currentCamera, direction === 'right');
            
            if (nextCamera) {
                // Update current camera and image
                currentCamera = nextCamera;
                updateFullImage();
            }
        };
        
        arrow.addEventListener('click', navigateImage);
        arrow.addEventListener('touchend', (e) => {
            e.preventDefault();
            navigateImage();
        });
        
        return arrow;
    };
    
    const leftArrow = createArrow('left');
    const rightArrow = createArrow('right');
    
    // Add keyboard navigation
    const handleKeyDown = (e) => {
        switch (e.key) {
            case 'ArrowLeft':
                const leftCamera = findNearestCameraInDirection(currentCamera, false);
                if (leftCamera) {
                    currentCamera = leftCamera;
                    updateFullImage();
                }
                break;
            case 'ArrowRight':
                const rightCamera = findNearestCameraInDirection(currentCamera, true);
                if (rightCamera) {
                    currentCamera = rightCamera;
                    updateFullImage();
                }
                break;
            case 'Escape':
                document.body.removeChild(overlay);
                document.removeEventListener('keydown', handleKeyDown);
                break;
            case '0':
            case 'r':
                // Reset zoom
                scale = 1;
                translateX = 0;
                translateY = 0;
                updateTransform();
                break;
        }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    
    // Clean up event listeners when overlay is removed
    overlay.addEventListener('remove', () => {
        document.removeEventListener('keydown', handleKeyDown);
        document.removeEventListener('mousemove', null);
        document.removeEventListener('mouseup', null);
    });
    
    // Assemble the components
    imgContainer.appendChild(fullImg);
    
    overlay.appendChild(closeBtn);
    overlay.appendChild(imgContainer);
    overlay.appendChild(leftArrow);
    overlay.appendChild(rightArrow);
    
    // Add instructions - show different instructions for touch devices
    const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    const instructionText = isTouchDevice ? 
        'Pinch to zoom, drag to pan when zoomed in, double-tap to reset zoom' : 
        'Use mouse wheel to zoom, drag to pan when zoomed in, double-click to reset zoom';
    
    const instructions = document.createElement('div');
    instructions.textContent = instructionText;
    instructions.style.position = 'absolute';
    instructions.style.bottom = '20px';
    instructions.style.left = '50%';
    instructions.style.transform = 'translateX(-50%)';
    instructions.style.color = 'white';
    instructions.style.background = 'rgba(0, 0, 0, 0.5)';
    instructions.style.padding = '8px 16px';
    instructions.style.borderRadius = '4px';
    instructions.style.fontSize = '14px';
    instructions.style.textAlign = 'center';
    instructions.style.maxWidth = '90%';
    overlay.appendChild(instructions);
    
    document.body.appendChild(overlay);
}