import DOMPurify from 'dompurify';
import { isString } from 'lodash';

DOMPurify.setConfig({
  ADD_ATTR: ['target'],
});

DOMPurify.addHook('afterSanitizeAttributes', function fixTarget(node) {
  // Fix elements with `target` attribute:
  // - allow only `target="_blank"
  // - add `rel="noopener noreferrer"` to prevent https://www.owasp.org/index.php/Reverse_Tabnabbing

  const target = node.getAttribute('target');
  if (isString(target) && target.toLowerCase() === '_blank') {
    node.setAttribute('rel', 'noopener noreferrer');
  } else {
    node.removeAttribute('target');
  }
});

export { DOMPurify };

export default DOMPurify.sanitize;
