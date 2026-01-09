// Override webpack public path for dynamic imports
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore - webpack magic variable
__webpack_public_path__ = '/static-files/';

// Bootstrapping asynchronously to avoid eager consumption of shared modules.
/* webpackMode: "eager" */
import('./bootstrap');
