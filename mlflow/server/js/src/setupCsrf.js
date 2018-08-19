import $ from 'jquery';
import cookie from 'cookie';

// To enable running behind applications that require CSRF tokens, we
// support parsing an optional "mlflow-csrf-token" cookie, which we will
// add as an 'X-CSRF-Token' header to all AJAX requests.
export const setupCsrf = () => {
  const csrfToken = getCsrfToken();
  $.ajaxSetup({
    beforeSend(xhr) {
      if (csrfToken) {
        xhr.setRequestHeader(CSRF_HEADER_NAME, csrfToken);
      }
    }
  });
};

export const getCsrfToken = () => {
  const parsedCookie = cookie.parse(document.cookie);
  return parsedCookie['mlflow-csrf-token'];
};

export const CSRF_HEADER_NAME = 'X-CSRF-Token';
