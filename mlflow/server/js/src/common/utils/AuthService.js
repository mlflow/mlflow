export class AuthService {
  token = null;
  toRedirectState = '';
  ssoLoginHeader = 'X-SSO-Login';
  ssoLogoutHeader = 'X-SSO-Logout';
  redirectUriRoute = 'oauth';

  constructor() {
    const savedToken = localStorage.getItem('token');
    if (savedToken !== null) {
      this.token = savedToken;
    }
    this.toRedirectState = window.location.pathname;
  }

  logOff() {
    this.setNewTokenInternal(null);
    const fullLogoutUrl = this.getFullLogoutUrl();
    if (fullLogoutUrl !== null) {
      window.location.assign(fullLogoutUrl);
    } else {
      window.location.reload();
    }
  }

  getFullLogoutUrl() {
    const logoutUrl = localStorage.getItem('logoutUrl');
    if (logoutUrl !== null) {
      return (
        logoutUrl +
        '&goto=' +
        encodeURIComponent(window.location.protocol + '//' + window.location.host + '/')
      );
    }
    return null;
  }

  getToken() {
    return this.token;
  }

  setNewToken(token) {
    this.setNewTokenInternal(token);
  }

  clearAuth() {
    this.setNewTokenInternal(null);
  }

  redirectUrl() {
    return window.location.protocol + '//' + window.location.host + '/' + this.redirectUriRoute;
  }

  /**
   * If can retrieve the SSO Url, intiate a redirect. else print an error
   * @param unauthorized the unauthorized http response
   */
  redirectToSsoIfPossible(unauthorized) {
    window.location.assign(this.getSsoUrl(unauthorized));
  }

  /**
   * If the SSO location header is set, store logout URL and return SSO URL. else print an error
   * @param unauthorized the unauthorized http response
   */
  getSsoUrl(unauthorized) {
    const loginUrl = unauthorized.getResponseHeader(this.ssoLoginHeader);
    if (loginUrl !== null) {
      const fullSsoUrl =
        loginUrl +
        '&redirect_uri=' +
        encodeURIComponent(this.redirectUrl()) +
        '&state=' +
        encodeURIComponent(this.toRedirectState);
      const logoutUrl = unauthorized.getResponseHeader(this.ssoLogoutHeader);
      if (logoutUrl !== null) {
        localStorage.setItem('logoutUrl', logoutUrl);
      }
      return fullSsoUrl;
    }
    throw new Error('Unauthorized error received, but no sso location provided');
  }

  setNewTokenInternal(token) {
    this.token = token;
    if (token !== null) {
      localStorage.setItem('token', token);
    } else {
      localStorage.removeItem('token');
    }
  }
}
