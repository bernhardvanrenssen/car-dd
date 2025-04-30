import { BoundingBox, Color, Mat4, Script, Vec3, Entity, StandardMaterial, Quat, BLEND_NONE, BLEND_NORMAL, LAYER_WORLD, Ray } from 'playcanvas';
import { CubicSpline } from 'spline';

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
        const blur = (progress) => blur("${Math.floor((100 - progress) * 0.4)}px");
        const element = document.getElementById('poster');
        element.style.backgroundImage = url("${url}");
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
        const events = ['wheel', 'pointerdown', 'contextmenu'];
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

    // Store references to all mesh objects for toggling
    const meshObjects = {
        debugObjects: [],
        cameraMarkers: []
    };

    // Function to create a button
    const createButton = (text, position, onClick) => {
        const button = document.createElement('button');
        button.textContent = text;
        button.style.position = 'absolute';
        button.style.zIndex = '1000';
        button.style.padding = '8px 16px';
        button.style.backgroundColor = '#007bff';
        button.style.color = 'white';
        button.style.border = 'none';
        button.style.borderRadius = '4px';
        button.style.cursor = 'pointer';
        if (position === 'top-left') {
            button.style.top = '10px';
            button.style.left = '10px';
        } else if (position === 'top-right') {
            button.style.top = '10px';
            button.style.right = '10px';
        } else if (position === 'bottom-left') {
            button.style.bottom = '10px';
            button.style.left = '10px';
        } else if (position === 'bottom-right') {
            button.style.bottom = '10px';
            button.style.right = '10px';
        }
        button.addEventListener('click', onClick);
        document.body.appendChild(button);
        return button;
    };

    // Function to create a debug box
    const createDebugBox = (app, position, color, scale) => {
        try {
            const box = new Entity(`DebugBox_${position.toString()}`);
            box.addComponent("model", { type: "box" });
            box.setPosition(position.x, position.y, position.z);
            box.setLocalScale(scale, scale, scale);
            const material = new StandardMaterial();
            material.diffuse = color;
            material.emissive = color;
            material.emissiveIntensity = 0.5;
            material.opacity = 1.0;
            material.blendType = BLEND_NORMAL;
            material.depthWrite = false;
            material.depthTest = true;
            material.cull = 0;
            material.update();
            if (box.model) {
                box.model.material = material;
                if (box.model.meshInstances && box.model.meshInstances.length > 0) {
                    box.model.meshInstances.forEach(meshInstance => {
                        meshInstance.layer = LAYER_WORLD + 1;
                        meshInstance.drawOrder = 9999;
                    });
                }
            }
            return box;
        } catch (err) {
            console.error(`Error creating debug box at ${position.toString()}:`, err);
            return null;
        }
    };

    // Function to add debug objects to the scene
    const addDebugObjects = (app) => {
        console.log("Adding debug objects to the scene");
        try {
            const debugParent = new Entity('DebugObjects');
            app.root.addChild(debugParent);
            const originBox = createDebugBox(app, new Vec3(0, 0, 0), new Color(1, 0, 1), 0.5);
            debugParent.addChild(originBox);
            meshObjects.debugObjects.push(originBox);
            const xAxisBox = createDebugBox(app, new Vec3(5, 0, 0), new Color(1, 0, 0), 0.3);
            const yAxisBox = createDebugBox(app, new Vec3(0, 5, 0), new Color(0, 1, 0), 0.3);
            const zAxisBox = createDebugBox(app, new Vec3(0, 0, 5), new Color(0, 0, 1), 0.3);
            debugParent.addChild(xAxisBox);
            debugParent.addChild(yAxisBox);
            debugParent.addChild(zAxisBox);
            meshObjects.debugObjects.push(xAxisBox, yAxisBox, zAxisBox);
            app.renderNextFrame = true;
            return debugParent;
        } catch (err) {
            console.error("Error adding debug objects:", err);
            return null;
        }
    };

    // Modified createCameraMarkers: now stores the original color on each marker.
    const createCameraMarkers = (app, cameraData) => {
        if (!cameraData || !Array.isArray(cameraData)) {
            console.log("No camera data provided");
            return;
        }
        console.log(`Creating ${cameraData.length} camera markers`);
        try {
            const existingMarkers = [];
            app.root.findByName('CameraMarkers', existingMarkers);
            existingMarkers.forEach(marker => {
                try {
                    if (marker && marker.parent) {
                        marker.parent.removeChild(marker);
                    }
                } catch (err) {
                    console.error("Error removing existing marker:", err);
                }
            });
            meshObjects.cameraMarkers = [];
            const markersParent = new Entity('CameraMarkers');
            app.root.addChild(markersParent);
            
            // Define fixed colors for clusters 0, 1, and 2.
            const clusterColors = [
                new Color(1, 1, 0), // Red for cluster 0
                new Color(0, 1, 0), // Green for cluster 1
                new Color(0, 0, 1)  // Blue for cluster 2
            ];
            
            cameraData.forEach((data, index) => {
                try {
                    const sphere = new Entity(`CameraMarker_${index}`);
                    sphere.addComponent("model", { type: "sphere" });
                    sphere.setLocalScale(0.2, 0.2, 0.2);
                    
                    // Determine the marker color based on the cluster.
                    const cluster = data.cluster;
                    const markerColor = (cluster !== undefined && cluster >= 0)
                        ? clusterColors[cluster % clusterColors.length]
                        : new Color(Math.random(), Math.random(), Math.random());
                        
                    // Save the original color for later reversion.
                    sphere.originalColor = markerColor.clone();
                    const material = new StandardMaterial();
                    material.diffuse = markerColor;
                    material.emissive = markerColor;
                    material.emissiveIntensity = 0.5;
                    material.opacity = 1.0;
                    material.blendType = BLEND_NORMAL;
                    material.depthWrite = false;
                    material.depthTest = true;
                    material.cull = 0;
                    material.update();
                    if (sphere.model) {
                        sphere.model.material = material;
                        if (sphere.model.meshInstances && sphere.model.meshInstances.length > 0) {
                            sphere.model.meshInstances.forEach(meshInstance => {
                                meshInstance.layer = LAYER_WORLD + 1;
                                meshInstance.drawOrder = 9999;
                            });
                        }
                    }
                    // Save the camera data on the marker.
                    sphere.cameraData = data;
                    sphere.setPosition(data.position.x, data.position.y, data.position.z);
                    markersParent.addChild(sphere);
                    meshObjects.cameraMarkers.push(sphere);
                    console.log(`Camera marker ${index} added at position: ${sphere.getPosition().toString()} with image: ${data.imageName} and cluster: ${data.cluster}`);
                } catch (err) {
                    console.error(`Error creating camera marker ${index}:`, err);
                }
            });
            app.renderNextFrame = true;
            return markersParent;
        } catch (err) {
            console.error("Error in createCameraMarkers:", err);
            return null;
        }
    };
    
    // Modified parse function for the converted file with 6 tokens per line.
    const parseCameraPositions = (text) => {
        const lines = text.split("\n").filter(line => line.trim() && line[0] !== "#");
        const cameras = [];
        lines.forEach(line => {
            const parts = line.trim().split(/\s+/);
            // If there are at least 7 tokens:
            // IMAGE_ID, Cx, Cy, Cz, CAMERA_ID, NAME (possibly with spaces), CLUSTER_LABEL
            if (parts.length >= 7) {
                const imageId = parseInt(parts[0], 10);
                const qw = parseFloat(parts[1]);
                const qx = parseFloat(parts[2]);
                const qy = parseFloat(parts[3]);
                const qz = parseFloat(parts[4]);
                const Cx = parseFloat(parts[5]);
                const Cy = parseFloat(parts[6]);
                const Cz = parseFloat(parts[7]);
                // Assume the file name occupies tokens 5 to second last token
                // const imageName = parts.slice(5, parts.length - 1).join(" ");
                // const cluster = parseInt(parts[parts.length - 1], 10);
                cameras.push({
                    imageId: parseInt(parts[0], 10),
                    position: { x: Cx, y: Cy, z: Cz },
                    quaternion: { qw, qx, qy, qz },
                    cameraId: parseInt(parts[8], 10),
                    imageName: parts.slice(9, parts.length - 1).join(" "),
                    cluster: parseInt(parts[parts.length - 1], 10)
                });
            } else if (parts.length >= 6) {
                // Fallback for older files without a cluster label.
                const imageId = parseInt(parts[0], 10);
                const Cx = parseFloat(parts[1]);
                const Cy = parseFloat(parts[2]);
                const Cz = parseFloat(parts[3]);
                const cameraId = parseInt(parts[4], 10);
                const imageName = parts.slice(5).join(" ");
                cameras.push({
                    imageId,
                    position: { x: Cx, y: Cy, z: Cz },
                    quaternion: { qw: 1, qx: 0, qy: 0, qz: 0 },
                    cameraId,
                    imageName,
                    cluster: -1  // or any default value for ungrouped images
                });
            }
        });
        return cameras;
    };
    

    // Function to load camera data from file
    const loadCameraData = () => {
        try {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.txt';
            input.style.display = 'none';
            document.body.appendChild(input);
            let fileInput = input;
            
            input.addEventListener('change', async (event) => {
                try {
                    if (event.target.files && event.target.files.length > 0) {
                        const file = event.target.files[0];
                        try {
                            const fileText = await file.text();
                            const cameras = parseCameraPositions(fileText);
                            console.log('Parsed cameras:', cameras);
                            // Create camera markers using existing functionality.
                            createCameraMarkers(app, cameras);
                            // Determine how many unique clusters exist.
                            const clusters = new Set(cameras.map(cam => cam.cluster));
                            window.numClusters = clusters.size;
                            console.log("Detected unique clusters:", window.numClusters);
                            const center = computeMarkersCenter();
                            window.meshObjects.cameraMarkers.forEach(marker => {
                                const pos = marker.getPosition();
                                // Calculate offset relative to the group's center.
                                const offset = pos.clone().sub(center);
                                // Flip the X and Y components of the offset.
                                const flippedOffset = new Vec3(-offset.x, -offset.y, offset.z);
                                // Set the new position as the center plus the flipped offset.
                                const newPos = center.clone().add(flippedOffset);
                                marker.setPosition(newPos.x, newPos.y, newPos.z);
                            });
                            app.renderNextFrame = true;
                        } catch (error) {
                            console.error("Error reading file:", error);
                        }
                    }
                    if (fileInput && fileInput.parentNode === document.body) {
                        document.body.removeChild(fileInput);
                    }
                    fileInput = null;
                } catch (err) {
                    console.error("Error in file input change handler:", err);
                }
            });
            
            const loadButton = createButton('Load Camera Data', 'bottom-right', () => {
                if (fileInput) {
                    fileInput.click();
                } else {
                    const newInput = document.createElement('input');
                    newInput.type = 'file';
                    newInput.accept = '.txt';
                    newInput.style.display = 'none';
                    document.body.appendChild(newInput);
                    fileInput = newInput;
                    newInput.addEventListener('change', async (event) => {
                        try {
                            if (event.target.files && event.target.files.length > 0) {
                                const file = event.target.files[0];
                                try {
                                    const fileText = await file.text();
                                    const cameras = parseCameraPositions(fileText);
                                    console.log('Parsed cameras:', cameras);
                                    createCameraMarkers(app, cameras);
                                    // Determine how many unique clusters exist.
                                    const clusters = new Set(cameras.map(cam => cam.cluster));
                                    window.numClusters = clusters.size;
                                    console.log("Detected unique clusters:", window.numClusters);
                                    const center = computeMarkersCenter();
                                    window.meshObjects.cameraMarkers.forEach(marker => {
                                        const pos = marker.getPosition();
                                        // Calculate offset relative to the group's center.
                                        const offset = pos.clone().sub(center);
                                        // Flip the X and Y components of the offset.
                                        const flippedOffset = new Vec3(-offset.x, -offset.y, offset.z);
                                        // Set the new position as the center plus the flipped offset.
                                        const newPos = center.clone().add(flippedOffset);
                                        marker.setPosition(newPos.x, newPos.y, newPos.z);
                                    });
                                    app.renderNextFrame = true;
                                } catch (error) {
                                    console.error("Error reading file:", error);
                                }
                            }
                            if (newInput && newInput.parentNode === document.body) {
                                document.body.removeChild(newInput);
                            }
                            fileInput = null;
                        } catch (err) {
                            console.error("Error in file input change handler:", err);
                        }
                    });
                    newInput.click();
                }
            });
            app.renderNextFrame = true;
        } catch (err) {
            console.error("Error in loadCameraData:", err);
        }
    };

    // Function to toggle mesh objects visibility
    const toggleMeshObjects = (visible) => {
        meshObjects.debugObjects.forEach(obj => {
            if (obj && obj.enabled !== undefined) {
                obj.enabled = visible;
            }
        });
        meshObjects.cameraMarkers.forEach(marker => {
            if (marker && marker.enabled !== undefined) {
                marker.enabled = visible;
            }
        });
        app.renderNextFrame = true;
    };

    // Add toggle button for mesh objects (top-left)
    let meshObjectsVisible = true;
    const toggleMeshButton = createButton('Toggle Mesh Objects', 'top-left', () => {
        meshObjectsVisible = !meshObjectsVisible;
        toggleMeshObjects(meshObjectsVisible);
        toggleMeshButton.textContent = meshObjectsVisible ? 'Hide Mesh Objects' : 'Show Mesh Objects';
    });
    toggleMeshButton.textContent = 'Hide Mesh Objects';

    // ---------------------------
    // NEW: Toggle for Camera Detection
    // ---------------------------
    let cameraDetectionEnabled = false;
    const cameraDetectButton = createButton('Enable Camera Detection', 'top-left', () => {
        cameraDetectionEnabled = !cameraDetectionEnabled;
        cameraDetectButton.textContent = cameraDetectionEnabled ? 'Disable Camera Detection' : 'Enable Camera Detection';
        // If detection is disabled, reset any highlighted marker
        if (!cameraDetectionEnabled && currentHighlightedMarker) {
            currentHighlightedMarker.model.material.diffuse = currentHighlightedMarker.originalColor;
            currentHighlightedMarker.model.material.emissive = currentHighlightedMarker.originalColor;
            currentHighlightedMarker.model.material.update();
            currentHighlightedMarker.isHighlighted = false;
            currentHighlightedMarker = null;
            app.renderNextFrame = true;
        }
    });
    cameraDetectButton.style.top = '50px';

    // ---------------------------
    // NEW: Button to Flip Imported Cameras in X and Y Axis
    // ---------------------------
    // const flipCamerasButton = createButton("Flip Cameras", "top-left", () => {
    //     // Iterate over each imported camera marker and flip the X and Y coordinates.
    //     const center = computeMarkersCenter();
    //     window.meshObjects.cameraMarkers.forEach(marker => {
    //         const pos = marker.getPosition();
    //         // Calculate offset relative to the group's center.
    //         const offset = pos.clone().sub(center);
    //         // Flip the X and Y components of the offset.
    //         const flippedOffset = new Vec3(-offset.x, -offset.y, offset.z);
    //         // Set the new position as the center plus the flipped offset.
    //         const newPos = center.clone().add(flippedOffset);
    //         marker.setPosition(newPos.x, newPos.y, newPos.z);
    //     });
    //     app.renderNextFrame = true;
    // });
    // flipCamerasButton.style.top = '90px';
    // flipCamerasButton.style.left = '10px';


    // ---------------------------
    // NEW: Function to parse key_images.txt file and return a Set of valid image IDs
    // ---------------------------
    function parseKeyImageFile(text) {
        const lines = text.split("\n");
        const keyIDs = new Set();
        lines.forEach(line => {
            if (!line.trim() || line.startsWith("#")) return;
            const tokens = line.trim().split(/\s+/);
            if (tokens.length >= 1) {
                const id = parseInt(tokens[0], 10);
                keyIDs.add(id);
            }
        });
        return keyIDs;
    }

    // ---------------------------
    // NEW: Function to prompt for key_images.txt file and fix the camera markers
    // ---------------------------
    function fixCameras() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.txt';
        input.style.display = 'none';
        document.body.appendChild(input);
        
        input.addEventListener('change', async (event) => {
            try {
                if (event.target.files && event.target.files.length > 0) {
                    const file = event.target.files[0];
                    const text = await file.text();
                    const keyIDs = parseKeyImageFile(text);
                    console.log("Key image IDs from selected file:", keyIDs);
                    // Remove markers that are not in the key set.
                    window.meshObjects.cameraMarkers.forEach(marker => {
                        if (!keyIDs.has(marker.cameraData.imageId)) {
                            if (marker.parent) {
                                marker.parent.removeChild(marker);
                            }
                        }
                    });
                    // Update the array to include only markers that remain.
                    window.meshObjects.cameraMarkers = window.meshObjects.cameraMarkers.filter(marker => marker.parent !== null);
                    app.renderNextFrame = true;
                }
            } catch (err) {
                console.error("Error processing key_images file:", err);
            } finally {
                document.body.removeChild(input);
            }
        });
        input.click();
    }
    // Create the Fix Cameras button below the rotation buttons.
    const fixCamerasButton = createButton("Fix Cameras", "bottom-left", fixCameras);
    fixCamerasButton.style.left = "290px";
    fixCamerasButton.style.bottom = "60px";

    // Function to create a visual origin gizmo at (0,0,0)
    function addOriginGizmo(app) {
        // X-axis: red bar
        const xAxis = new Entity("gizmo_x");
        xAxis.addComponent("model", { type: "box" });
        // Create a thin box along X: length of 1, thickness of 0.02 in Y and Z.
        xAxis.setLocalScale(1, 0.02, 0.02);
        // Position the box so that its left end is at the origin (centered at 0.5,0,0)
        xAxis.setPosition(0.5, 0, 0);
        const xMaterial = new StandardMaterial();
        xMaterial.diffuse = new Color(1, 0, 0);
        xMaterial.emissive = new Color(1, 0, 0);
        xMaterial.update();
        xAxis.model.material = xMaterial;

        // Y-axis: green bar
        const yAxis = new Entity("gizmo_y");
        yAxis.addComponent("model", { type: "box" });
        // Thin box along Y: length 1, thickness 0.02 in X and Z.
        yAxis.setLocalScale(0.02, 1, 0.02);
        yAxis.setPosition(0, 0.5, 0);
        const yMaterial = new StandardMaterial();
        yMaterial.diffuse = new Color(0, 1, 0);
        yMaterial.emissive = new Color(0, 1, 0);
        yMaterial.update();
        yAxis.model.material = yMaterial;

        // Z-axis: blue bar
        const zAxis = new Entity("gizmo_z");
        zAxis.addComponent("model", { type: "box" });
        // Thin box along Z: length 1, thickness 0.02 in X and Y.
        zAxis.setLocalScale(0.02, 0.02, 1);
        zAxis.setPosition(0, 0, 0.5);
        const zMaterial = new StandardMaterial();
        zMaterial.diffuse = new Color(0, 0, 1);
        zMaterial.emissive = new Color(0, 0, 1);
        zMaterial.update();
        zAxis.model.material = zMaterial;

        // Optional: a small sphere to emphasize the true origin at (0,0,0)
        const originSphere = new Entity("gizmo_origin");
        originSphere.addComponent("model", { type: "sphere" });
        // A small sphere (adjust scale as needed)
        originSphere.setLocalScale(0.05, 0.05, 0.05);
        originSphere.setPosition(0, 0, 0);
        const sphereMaterial = new StandardMaterial();
        // Here we use yellow to highlight the exact origin
        sphereMaterial.diffuse = new Color(1, 1, 0);
        sphereMaterial.emissive = new Color(1, 1, 0);
        sphereMaterial.update();
        originSphere.model.material = sphereMaterial;

        // Create a parent entity for the gizmo and add all parts to it.
        const originGizmo = new Entity("originGizmo");
        originGizmo.addChild(xAxis);
        originGizmo.addChild(yAxis);
        originGizmo.addChild(zAxis);
        originGizmo.addChild(originSphere);

        // Finally, add the gizmo to the scene.
        app.root.addChild(originGizmo);
    }

    addOriginGizmo(app);


    // Load camera data
    loadCameraData();

    addExportButton(app);

    camera.camera.clearColor = new Color(settings.background.color);
    camera.camera.fov = settings.camera.fov;
    camera.script.create(FrameScene, { properties: { settings } });

    // Update loading indicator
    const assets = app.assets.filter(asset => asset.type === 'gsplat');
    if (assets.length > 0) {
        const asset = assets[0];
        const loadingText = document.getElementById('loadingText');
        const loadingBar = document.getElementById('loadingBar');
        asset.on('progress', (received, length) => {
            const v = (Math.min(1, received / length) * 100).toFixed(0);
            loadingText.textContent = "${v}%";
            loadingBar.style.backgroundImage = 'linear-gradient(90deg, #F60 0%, #F60 ' + v + '%, white ' + v + '%, white 100%)';
            poster?.progress(v);
        });
    }

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
    // Hide UI
    if (params.noui) {
        dom.buttonContainer.classList.add('hidden');
    }

    window.meshObjects = meshObjects;

    // ---------------------------
    // NEW: Highlight Closest Camera Marker on Mouse Move
    // ---------------------------
    const canvas = document.querySelector('canvas');
    let currentHighlightedMarker = null;
    if (!canvas) {
        console.error("Canvas element not found.");
    } else {
        // ——— Replace your existing mousemove handler with this block ———
        canvas.addEventListener('mousemove', (event) => {
            if (!cameraDetectionEnabled) return;
        
            // 1) compute mouseX, mouseY
            const rect   = canvas.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;
        
            // 2) build pick‐ray
            const nearPoint = new Vec3();
            const farPoint  = new Vec3();
            camera.camera.screenToWorld(mouseX, mouseY, camera.camera.nearClip, nearPoint);
            camera.camera.screenToWorld(mouseX, mouseY, camera.camera.farClip,  farPoint);
            const direction = farPoint.sub(nearPoint).normalize();
            const ray       = new Ray(nearPoint, direction);
        
            // 3) intersect the AABB
            const gs = app.root.findComponent('gsplat');
            if (!gs?.instance?.meshInstance?.aabb) {
                if (currentHighlightedMarker) {
                    const m = currentHighlightedMarker.model.material;
                    m.diffuse  = currentHighlightedMarker.originalColor;
                    m.emissive = currentHighlightedMarker.originalColor;
                    m.update();
                    currentHighlightedMarker.isHighlighted = false;
                    currentHighlightedMarker = null;
                    app.renderNextFrame = true;
                }
                return;
            }
            const tNear = intersectAABB(ray, gs.instance.meshInstance.aabb);
            if (tNear === null) {
                if (currentHighlightedMarker) {
                    const m = currentHighlightedMarker.model.material;
                    m.diffuse  = currentHighlightedMarker.originalColor;
                    m.emissive = currentHighlightedMarker.originalColor;
                    m.update();
                    currentHighlightedMarker.isHighlighted = false;
                    currentHighlightedMarker = null;
                    app.renderNextFrame = true;
                }
                return;
            }
        
            // 4) compute hoverPoint on that AABB face
            const hoverPoint = direction.mulScalar(tNear).add(nearPoint);
        
            // 5) move your purple debug sphere
            if (!window.debugSphere) {
                window.debugSphere = new Entity('debugSphere');
                window.debugSphere.addComponent('model', { type: 'sphere' });
                window.debugSphere.setLocalScale(0.1, 0.1, 0.1);
                const mat = new StandardMaterial();
                mat.diffuse           = new Color(0.5, 0, 0.5);
                mat.emissive          = new Color(0.5, 0, 0.5);
                mat.emissiveIntensity = 0.8;
                mat.depthTest         = false;
                mat.depthWrite        = false;
                mat.update();
                window.debugSphere.model.material = mat;
                window.debugSphere.model.meshInstances.forEach(mi => {
                    mi.layer     = LAYER_WORLD + 1;
                    mi.drawOrder = 9999;
                });
                app.root.addChild(window.debugSphere);
            }
            window.debugSphere.setPosition(hoverPoint.x, hoverPoint.y, hoverPoint.z);
        
            // 6) find closest by weighted‐distance alone
            const horizontalWeight = 0.05;
            let   minDist = Infinity;
            let   best   = null;
            window.meshObjects.cameraMarkers.forEach(marker => {
                const mp = marker.getPosition();
                const dx = mp.x - hoverPoint.x;
                const dy = mp.y - hoverPoint.y;
                const dz = mp.z - hoverPoint.z;
                const dist = Math.sqrt(
                    horizontalWeight * (dx*dx + dz*dz) + (dy*dy)
                );
                if (dist < minDist) {
                    minDist = dist;
                    best    = marker;
                }
            });
        
            // 7) highlight only that one
            window.meshObjects.cameraMarkers.forEach(marker => {
                const m = marker.model.material;
                if (marker === best) {
                    if (!marker.isHighlighted) {
                        m.diffuse   = new Color(1, 0, 0);
                        m.emissive  = new Color(1, 0, 0);
                        m.update();
                        marker.isHighlighted = true;
                        currentHighlightedMarker = marker;
                    }
                } else if (marker.isHighlighted) {
                    m.diffuse   = marker.originalColor;
                    m.emissive  = marker.originalColor;
                    m.update();
                    marker.isHighlighted = false;
                }
            });
        
            app.renderNextFrame = true;
        });
        
        canvas.addEventListener('click', (event) => {
            if (!cameraDetectionEnabled) return;
        
            // 1) mouse→canvas coords
            const rect = canvas.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;
        
            // 2) build pick ray
            const nearPoint = new Vec3();
            const farPoint = new Vec3();
            camera.camera.screenToWorld(mouseX, mouseY, camera.camera.nearClip, nearPoint);
            camera.camera.screenToWorld(mouseX, mouseY, camera.camera.farClip, farPoint);
            const direction = farPoint.sub(nearPoint).normalize();
            const ray = new Ray(nearPoint, direction);
        
            // 3) intersect AABB
            const gs = app.root.findComponent('gsplat');
            if (!gs?.instance?.meshInstance?.aabb) return;
            const tNear = intersectAABB(ray, gs.instance.meshInstance.aabb);
            
            // CHANGE: Return early if no intersection is found
            if (tNear === null) {
                // Remove any existing popup when clicking empty space
                const old = document.getElementById('cameraPopup');
                if (old) old.remove();
                return; // Don't render a frame when clicking on empty space
            }
        
            // 4) compute clickPoint
            const clickPoint = direction.mulScalar(tNear).add(nearPoint);
        
            // 5) move debug sphere
            if (window.debugSphere) {
                window.debugSphere.setPosition(clickPoint.x, clickPoint.y, clickPoint.z);
            }
        
            // 6) pick purely on distance
            const horizontalWeight = 0.05;
            let minDist = Infinity;
            let best = null;
            window.meshObjects.cameraMarkers.forEach(marker => {
                const mp = marker.getPosition();
                const dx = mp.x - clickPoint.x;
                const dy = mp.y - clickPoint.y;
                const dz = mp.z - clickPoint.z;
                const dist = Math.sqrt(
                    horizontalWeight * (dx*dx + dz*dz) + (dy*dy)
                );
                if (dist < minDist) {
                    minDist = dist;
                    best = marker;
                }
            });
        
            // CODE FOR THE POPUP IMAGE
            if (best) {
                console.log(
                    "Selected camera marker:", best.name,
                    "with image:", best.cameraData.imageName
                );

                // 1. remove old popup (if any)
                const old = document.getElementById('cameraPopup');
                if (old) old.remove();

                // 2. build new <img> container with expand button
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
                
                // Add the thumbnail image
                const img = document.createElement('img');
                img.src = `images/${best.cameraData.imageName}`;
                img.style.width = '100px';
                img.style.height = 'auto';
                img.style.pointerEvents = 'none';
                container.appendChild(img);
                
                // Add expand button
                const expandBtn = document.createElement('button');
                expandBtn.textContent = 'View Full Size';
                expandBtn.style.padding = '4px';
                expandBtn.style.margin = '4px';
                expandBtn.style.cursor = 'pointer';
                expandBtn.style.background = '#007bff';
                expandBtn.style.color = 'white';
                expandBtn.style.border = 'none';
                expandBtn.style.borderRadius = '4px';
                
                // Function to find nearest camera marker in a direction
                const findNearestMarkerInDirection = (currentMarker, isRightDirection) => {
                    const currentPos = currentMarker.getPosition();
                    // Get all markers except the current one
                    const otherMarkers = window.meshObjects.cameraMarkers.filter(m => m !== currentMarker);
                    
                    if (otherMarkers.length === 0) return null;
                    
                    // Sort markers by angular position relative to the center
                    const center = computeMarkersCenter();
                    
                    // Calculate current marker's angle from center
                    const currentDx = currentPos.x - center.x;
                    const currentDz = currentPos.z - center.z;
                    const currentAngle = Math.atan2(currentDz, currentDx);
                    
                    // Calculate angles for all other markers
                    const markersWithAngles = otherMarkers.map(marker => {
                        const pos = marker.getPosition();
                        const dx = pos.x - center.x;
                        const dz = pos.z - center.z;
                        let angle = Math.atan2(dz, dx);
                        
                        // Calculate angular difference (considering circular wrap-around)
                        let diff = angle - currentAngle;
                        if (diff > Math.PI) diff -= 2 * Math.PI;
                        if (diff < -Math.PI) diff += 2 * Math.PI;
                        
                        return {
                            marker,
                            angle,
                            diff
                        };
                    });
                    
                    // Filter markers based on direction (positive diff is right/clockwise)
                    const filteredMarkers = markersWithAngles.filter(m => 
                        isRightDirection ? m.diff > 0 : m.diff < 0
                    );
                    
                    // If no markers in desired direction, wrap around (take the furthest in opposite direction)
                    if (filteredMarkers.length === 0) {
                        const oppositeDirMarkers = markersWithAngles.filter(m => 
                            isRightDirection ? m.diff < 0 : m.diff > 0
                        );
                        
                        if (oppositeDirMarkers.length === 0) return null;
                        
                        // Sort by abs angle difference in descending order (furthest first)
                        oppositeDirMarkers.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));
                        return oppositeDirMarkers[0].marker;
                    }
                    
                    // Sort by abs angle difference in ascending order (nearest first)
                    filteredMarkers.sort((a, b) => Math.abs(a.diff) - Math.abs(b.diff));
                    return filteredMarkers[0].marker;
                };
                
                // Add click handler for the expand button
                expandBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    
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
                    
                    // Create image container for zoom functionality
                    const imgContainer = document.createElement('div');
                    imgContainer.style.position = 'relative';
                    imgContainer.style.width = '80%';
                    imgContainer.style.height = '80%';
                    imgContainer.style.display = 'flex';
                    imgContainer.style.justifyContent = 'center';
                    imgContainer.style.alignItems = 'center';
                    imgContainer.style.overflow = 'hidden';
                    
                    // Current marker reference for navigation
                    let currentMarker = best;
                    
                    // Create image wrapper for panning
                    const imgWrapper = document.createElement('div');
                    imgWrapper.style.position = 'relative';
                    imgWrapper.style.overflow = 'hidden';
                    imgWrapper.style.width = '100%';
                    imgWrapper.style.height = '100%';
                    imgWrapper.style.display = 'flex';
                    imgWrapper.style.justifyContent = 'center';
                    imgWrapper.style.alignItems = 'center';
                    
                    // Create fullsize image
                    const fullImg = document.createElement('img');
                    fullImg.src = `images/${currentMarker.cameraData.imageName}`;
                    fullImg.style.maxWidth = '100%';
                    fullImg.style.maxHeight = '100%';
                    fullImg.style.objectFit = 'contain';
                    fullImg.style.transformOrigin = 'center center';
                    
                    // Zoom and pan variables
                    let scale = 1;
                    let posX = 0;
                    let posY = 0;
                    let startPosX = 0;
                    let startPosY = 0;
                    let startClientX = 0;
                    let startClientY = 0;
                    let isDragging = false;
                    
                    // Function to update transform
                    const updateTransform = () => {
                        fullImg.style.transform = `translate(${posX}px, ${posY}px) scale(${scale})`;
                    };
                    
                    // Function to update image when changing
                    const updateFullImage = () => {
                        fullImg.src = `images/${currentMarker.cameraData.imageName}`;
                        // Reset zoom and position
                        scale = 1;
                        posX = 0;
                        posY = 0;
                        updateTransform();
                    };
                    
                    // Mouse wheel zoom
                    imgWrapper.addEventListener('wheel', (e) => {
                        e.preventDefault();
                        
                        // Get mouse position relative to image
                        const rect = fullImg.getBoundingClientRect();
                        const mouseX = e.clientX - rect.left;
                        const mouseY = e.clientY - rect.top;
                        
                        // Calculate zoom change
                        const delta = e.deltaY > 0 ? 0.9 : 1.1;
                        const newScale = Math.max(1, Math.min(5, scale * delta));
                        
                        if (newScale !== scale) {
                            // Calculate new position to zoom at mouse point
                            const scaleRatio = newScale / scale;
                            const newPosX = mouseX - (mouseX - posX) * scaleRatio;
                            const newPosY = mouseY - (mouseY - posY) * scaleRatio;
                            
                            // Update state
                            scale = newScale;
                            
                            // Only allow panning when zoomed in
                            if (scale > 1) {
                                posX = newPosX;
                                posY = newPosY;
                            } else {
                                posX = 0;
                                posY = 0;
                            }
                            
                            updateTransform();
                        }
                    });
                    
                    // Mouse down for panning
                    imgWrapper.addEventListener('mousedown', (e) => {
                        // Only allow panning when zoomed in
                        if (scale <= 1) return;
                        
                        isDragging = true;
                        startClientX = e.clientX;
                        startClientY = e.clientY;
                        startPosX = posX;
                        startPosY = posY;
                        imgWrapper.style.cursor = 'grabbing';
                    });
                    
                    // Mouse move for panning
                    document.addEventListener('mousemove', (e) => {
                        if (!isDragging) return;
                        
                        // Calculate new position
                        posX = startPosX + (e.clientX - startClientX);
                        posY = startPosY + (e.clientY - startClientY);
                        
                        updateTransform();
                    });
                    
                    // Mouse up to end panning
                    document.addEventListener('mouseup', () => {
                        isDragging = false;
                        imgWrapper.style.cursor = scale > 1 ? 'grab' : 'default';
                    });
                    
                    // Double click to reset zoom
                    imgWrapper.addEventListener('dblclick', () => {
                        scale = 1;
                        posX = 0;
                        posY = 0;
                        updateTransform();
                        imgWrapper.style.cursor = 'default';
                    });
                    
                    // Fix: Completely revise the findNearestMarkerInDirection function
                    const findNearestMarkerInDirection = (currentMarker, isRightDirection) => {
                        const currentPos = currentMarker.getPosition();
                        // Get all markers except the current one
                        const otherMarkers = window.meshObjects.cameraMarkers.filter(m => m !== currentMarker);
                        
                        if (otherMarkers.length === 0) return null;
                        
                        // Get camera viewing direction (assuming forward is negative Z in world space)
                        // Use the center of all markers as the central reference point
                        const center = computeMarkersCenter();
                        
                        // Vector from center to current marker
                        const markerVec = new Vec3(
                            currentPos.x - center.x,
                            0, // Ignore Y for horizontal navigation
                            currentPos.z - center.z
                        ).normalize();
                        
                        // Compute a right vector (perpendicular to marker vector)
                        // Assuming Y is up, cross with world up to get right vector
                        const worldUp = new Vec3(0, 1, 0);
                        const rightVec = new Vec3().cross(worldUp, markerVec).normalize();
                        
                        // For each marker, compute how far it is in the right or left direction
                        const markersWithDirectionality = otherMarkers.map(marker => {
                            const pos = marker.getPosition();
                            
                            // Vector from current marker to other marker
                            const toMarkerVec = new Vec3(
                                pos.x - currentPos.x,
                                0, // Ignore Y for horizontal navigation
                                pos.z - currentPos.z
                            );
                            
                            // If toMarkerVec is zero length, skip this marker
                            if (toMarkerVec.length() < 0.001) return { marker, score: -Infinity };
                            
                            toMarkerVec.normalize();
                            
                            // Dot product with right vector gives how far right the marker is
                            // Positive is right, negative is left
                            const rightAmount = rightVec.dot(toMarkerVec);
                            
                            // Only consider markers that are in the correct half-space
                            const forwardAmount = markerVec.dot(toMarkerVec);
                            
                            // Distance factor (prefer closer markers)
                            const distance = Math.sqrt(
                                Math.pow(pos.x - currentPos.x, 2) +
                                Math.pow(pos.z - currentPos.z, 2)
                            );
                            
                            // Score combines direction and distance
                            // Higher score = better match (more in the right direction and closer)
                            const directionScore = isRightDirection ? rightAmount : -rightAmount;
                            
                            // We want markers that are:
                            // 1. In the correct direction (right or left)
                            // 2. More forward than backward
                            // 3. Closer rather than farther
                            const score = directionScore * 3 + forwardAmount * 2 - (distance * 0.1);
                            
                            return { marker, score };
                        });
                        
                        // Find the marker with the highest score
                        markersWithDirectionality.sort((a, b) => b.score - a.score);
                        
                        // Return the best marker, or null if none found
                        return markersWithDirectionality.length > 0 && 
                            markersWithDirectionality[0].score > -1 ? 
                            markersWithDirectionality[0].marker : null;
                    };
                    
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
                        
                        arrow.addEventListener('click', () => {
                            // Left means go to marker on the left, right means go to marker on the right
                            const nextMarker = findNearestMarkerInDirection(currentMarker, direction === 'right');
                            
                            if (nextMarker) {
                                // Also highlight the selected marker in the 3D view
                                window.meshObjects.cameraMarkers.forEach(marker => {
                                    if (marker.isHighlighted && marker !== nextMarker) {
                                        const m = marker.model.material;
                                        m.diffuse = marker.originalColor;
                                        m.emissive = marker.originalColor;
                                        m.update();
                                        marker.isHighlighted = false;
                                    }
                                });
                                
                                // Highlight the new marker
                                const m = nextMarker.model.material;
                                m.diffuse = new Color(1, 0, 0);
                                m.emissive = new Color(1, 0, 0);
                                m.update();
                                nextMarker.isHighlighted = true;
                                
                                // Update current marker and image
                                currentMarker = nextMarker;
                                updateFullImage();
                                
                                // Force a render to show the highlighted marker
                                app.renderNextFrame = true;
                            }
                        });
                        
                        return arrow;
                    };
                    
                    const leftArrow = createArrow('left');
                    const rightArrow = createArrow('right');
                    
                    // Add keyboard navigation
                    const handleKeyDown = (e) => {
                        switch (e.key) {
                            case 'ArrowLeft':
                                const leftMarker = findNearestMarkerInDirection(currentMarker, false);
                                if (leftMarker) {
                                    // Update highlighted marker
                                    window.meshObjects.cameraMarkers.forEach(marker => {
                                        if (marker.isHighlighted && marker !== leftMarker) {
                                            const m = marker.model.material;
                                            m.diffuse = marker.originalColor;
                                            m.emissive = marker.originalColor;
                                            m.update();
                                            marker.isHighlighted = false;
                                        }
                                    });
                                    
                                    const m = leftMarker.model.material;
                                    m.diffuse = new Color(1, 0, 0);
                                    m.emissive = new Color(1, 0, 0);
                                    m.update();
                                    leftMarker.isHighlighted = true;
                                    
                                    currentMarker = leftMarker;
                                    updateFullImage();
                                    app.renderNextFrame = true;
                                }
                                break;
                            case 'ArrowRight':
                                const rightMarker = findNearestMarkerInDirection(currentMarker, true);
                                if (rightMarker) {
                                    // Update highlighted marker
                                    window.meshObjects.cameraMarkers.forEach(marker => {
                                        if (marker.isHighlighted && marker !== rightMarker) {
                                            const m = marker.model.material;
                                            m.diffuse = marker.originalColor;
                                            m.emissive = marker.originalColor;
                                            m.update();
                                            marker.isHighlighted = false;
                                        }
                                    });
                                    
                                    const m = rightMarker.model.material;
                                    m.diffuse = new Color(1, 0, 0);
                                    m.emissive = new Color(1, 0, 0);
                                    m.update();
                                    rightMarker.isHighlighted = true;
                                    
                                    currentMarker = rightMarker;
                                    updateFullImage();
                                    app.renderNextFrame = true;
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
                                posX = 0;
                                posY = 0;
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
                    imgWrapper.appendChild(fullImg);
                    imgContainer.appendChild(imgWrapper);
                    
                    overlay.appendChild(closeBtn);
                    overlay.appendChild(imgContainer);
                    overlay.appendChild(leftArrow);
                    overlay.appendChild(rightArrow);
                    
                    // Add zoom instructions
                    const instructions = document.createElement('div');
                    instructions.textContent = 'Use mouse wheel to zoom, drag to pan when zoomed in, double-click to reset zoom';
                    instructions.style.position = 'absolute';
                    instructions.style.bottom = '20px';
                    instructions.style.left = '50%';
                    instructions.style.transform = 'translateX(-50%)';
                    instructions.style.color = 'white';
                    instructions.style.background = 'rgba(0, 0, 0, 0.5)';
                    instructions.style.padding = '8px 16px';
                    instructions.style.borderRadius = '4px';
                    instructions.style.fontSize = '14px';
                    overlay.appendChild(instructions);
                    
                    document.body.appendChild(overlay);
                });
                
                container.appendChild(expandBtn);
                document.body.appendChild(container);
                
                // Only request a render frame if we actually found a marker
                app.renderNextFrame = true;
            }
        });
    }

    // ---------------------------
    // ROTATION FUNCTIONALITY FOR CAMERA MARKERS
    // ---------------------------
    function computeMarkersCenter() {
        const markers = window.meshObjects.cameraMarkers;
        if (!markers || markers.length === 0) return new Vec3(0, 0, 0);
        const center = new Vec3(0, 0, 0);
        markers.forEach(marker => {
            center.add(marker.getPosition());
        });
        center.scale(1 / markers.length);
        return center;
    }
    
    function rotateCameraMarkers(angle) {
        const markers = window.meshObjects.cameraMarkers;
        if (!markers || markers.length === 0) return;
        const center = computeMarkersCenter();
        markers.forEach(marker => {
            const pos = marker.getPosition();
            const offset = pos.clone().sub(center);
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            const newX = offset.x * cos - offset.z * sin;
            const newZ = offset.x * sin + offset.z * cos;
            const newPos = new Vec3(newX, offset.y, newZ).add(center);
            marker.setPosition(newPos.x, newPos.y, newPos.z);
        });
        app.renderNextFrame = true;
    }

    function scaleCameraMarkers(scaleFactor) {
        const markers = window.meshObjects.cameraMarkers;
        if (!markers || markers.length === 0) return;
        const center = computeMarkersCenter();
        markers.forEach(marker => {
            const pos = marker.getPosition(); // current position as a Vec3
            const offset = pos.clone().sub(center); // vector from center to marker
            // Compute the new position: center + (offset * scaleFactor)
            const newOffset = offset.clone().mulScalar(scaleFactor);
            const newPos = center.clone().add(newOffset);
            marker.setPosition(newPos.x, newPos.y, newPos.z);
        });
        app.renderNextFrame = true;
    }
    
    const rotateLeftButton = createButton("Rotate Left", "bottom-left", () => {
        const angle = -10 * Math.PI / 180;
        rotateCameraMarkers(angle);
    });
    rotateLeftButton.style.left = "10px";
    rotateLeftButton.style.bottom = "60px";
    
    const rotateRightButton = createButton("Rotate Right", "bottom-left", () => {
        const angle = 10 * Math.PI / 180;
        rotateCameraMarkers(angle);
    });
    rotateRightButton.style.left = "150px";
    rotateRightButton.style.bottom = "60px";



    const scaleUpButton = createButton("Scale +", "bottom-left", () => {
        // Increase scale by 10%
        scaleCameraMarkers(1.1);
    });
    scaleUpButton.style.left = "10px";
    scaleUpButton.style.bottom = "10px";  // Adjust as needed so they are below the rotate buttons
    
    const scaleDownButton = createButton("Scale -", "bottom-left", () => {
        // Decrease scale by 10%
        scaleCameraMarkers(0.9);
    });
    scaleDownButton.style.left = "150px";
    scaleDownButton.style.bottom = "10px";  // Adjust as needed so they are below the rotate buttons
    
    // Function to export camera positions to a JSON file
    function exportCameraPositions() {
        try {
            // Get all camera markers
            const markers = window.meshObjects.cameraMarkers;
            if (!markers || markers.length === 0) {
                console.error("No camera markers to export");
                alert("No camera markers to export!");
                return;
            }

            // Build the export data structure
            const exportData = markers.map(marker => {
                const pos = marker.getPosition();
                const cameraData = marker.cameraData;
                
                return {
                    imageId: cameraData.imageId,
                    position: {
                        x: pos.x,
                        y: pos.y,
                        z: pos.z
                    },
                    quaternion: cameraData.quaternion || { qw: 1, qx: 0, qy: 0, qz: 0 },
                    cameraId: cameraData.cameraId,
                    imageName: cameraData.imageName,
                    cluster: cameraData.cluster || -1
                };
            });

            // Convert to JSON string with nice formatting
            const jsonString = JSON.stringify(exportData, null, 2);
            
            // Create a blob and download link
            const blob = new Blob([jsonString], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            // Create and trigger download
            const a = document.createElement('a');
            a.href = url;
            a.download = 'camera_positions.json';
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
            
            console.log(`Exported ${exportData.length} camera positions`);
        } catch (err) {
            console.error("Error exporting camera positions:", err);
            alert("Error exporting camera positions: " + err.message);
        }
    }

    // Add export button to the UI (JSON only)
    function addExportButton(app) {
        // Export as JSON button
        const exportJsonButton = document.createElement('button');
        exportJsonButton.textContent = 'Export Cameras';
        exportJsonButton.style.position = 'absolute';
        exportJsonButton.style.zIndex = '1000';
        exportJsonButton.style.padding = '8px 16px';
        exportJsonButton.style.backgroundColor = '#007bff';
        exportJsonButton.style.color = 'white';
        exportJsonButton.style.border = 'none';
        exportJsonButton.style.borderRadius = '4px';
        exportJsonButton.style.cursor = 'pointer';
        exportJsonButton.style.bottom = '60px';
        exportJsonButton.style.right = '10px';
        exportJsonButton.addEventListener('click', exportCameraPositions);
        document.body.appendChild(exportJsonButton);
        
        app.renderNextFrame = true;
    }

});