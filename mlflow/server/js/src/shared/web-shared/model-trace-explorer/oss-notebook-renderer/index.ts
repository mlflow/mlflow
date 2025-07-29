// Override webpack public path for dynamic imports
declare const __webpack_public_path__ = '/static-files/';

// Bootstrapping asynchronously to avoid eager consumption of shared modules.
/* webpackMode: "eager" */
import('./bootstrap');
