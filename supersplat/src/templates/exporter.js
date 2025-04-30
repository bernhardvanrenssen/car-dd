// Wait for the app and window.meshObjects to be initialized
document.addEventListener('DOMContentLoaded', () => {
  const checkInterval = setInterval(() => {
    if (window.meshObjects) {
      clearInterval(checkInterval);
      // Add all the export functions here
    }
  }, 100);
});