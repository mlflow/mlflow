import $ from 'jquery';
import cookie from 'cookie';

// To enable running behind applications that require specific headers
// to be set during HTTP requests (e.g., CSRF tokens), we support parsing
// a set of cookies with a key prefix of "mlflow-request-header-$HeaderName",
// which will be added as an HTTP header to all AJAX requests.
export const setupAjaxHeaders = () => {
  const requestHeaders = getRequestHeaders();
  $.ajaxSetup({
    beforeSend(xhr) {
      if (requestHeaders) {
        for (const headerKey in requestHeaders) {
          xhr.setRequestHeader(headerKey, requestHeaders[headerKey]);
        }
      }
    }
  });
};

export const getRequestHeaders = () => {
  const headerCookiePrefix = "mlflow-request-header-";
  const parsedCookie = cookie.parse(document.cookie);
  console.log(parsedCookie);
  const headers = {}
  for (const cookieName in parsedCookie) {
    if (cookieName.startsWith(headerCookiePrefix)) {
      headers[cookieName.substring(headerCookiePrefix.length)] = parsedCookie[cookieName];
    }
  }
  return headers;
};
