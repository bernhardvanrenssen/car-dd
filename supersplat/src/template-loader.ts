// src/template-loader.ts

// Import the template files
import viewerDevHtml from './templates/viewer-dev.html';
import viewerClientHtml from './templates/viewer-client.html';
import indexCss from './templates/index.css';
// Import your JavaScript files
import indexDevJs from './templates/index-dev.js';
import indexClientJs from './templates/index-client.js';
import splineJs from './templates/spline.js';

/**
 * Load the HTML template based on version type
 */
export function loadHtmlTemplate(versionType: 'dev' | 'client'): string {
    return versionType === 'dev' ? viewerDevHtml : viewerClientHtml;
}

/**
 * Load the JavaScript template based on version type
 */
export function loadJsTemplate(versionType: 'dev' | 'client'): string {
    return versionType === 'dev' ? indexDevJs : indexClientJs;
}

/**
 * Get CSS template (common for both versions)
 */
export function getCssTemplate(): string {
    return indexCss;
}

/**
 * Get Spline.js template (common for both versions)
 */
export function getSplineJs(): string {
    return splineJs;
}