import $ from 'jquery';
import cookie from 'cookie';

// To enable running behind applications that require specific headers
// to be set during HTTP requests (e.g., CSRF tokens), we support parsing
// a set of cookies with a key prefix of "mlflow-request-header-$HeaderName",
// which will be added as an HTTP header to all AJAX requests.
export const setupAjaxHeaders = () => {
  const requestHeaders = getRequestHeaders(document.cookie);
  $(document).ajaxSend((event, jqXHR) => {
    if (requestHeaders) {
      for (const [headerKey, headerValue] of Object.entries(requestHeaders)) {
        jqXHR.setRequestHeader(headerKey, headerValue);
      }
    }
  });
};

export const getRequestHeaders = (documentCookie) => {
  const headerCookiePrefix = 'mlflow-request-header-';
  const parsedCookie = cookie.parse(documentCookie);
  const headers = {};
  for (const cookieName in parsedCookie) {
    if (cookieName.startsWith(headerCookiePrefix)) {
      headers[cookieName.substring(headerCookiePrefix.length)] = parsedCookie[cookieName];
    }
  }
  return headers;
};
