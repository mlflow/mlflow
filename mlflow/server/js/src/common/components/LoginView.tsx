import React, { useEffect, useState } from 'react';
// import Keycloak from 'keycloak-js';
import keycloak from '../../keycloak-config';
import { MutatingDots } from 'react-loader-spinner';
const LoginView: React.FC = () => {
  const [keycloakAuthenticated, setKeycloakAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  // console.log("process.env['REACT_APP_WEB_BASE_URL']", process.env['REACT_APP_WEB_BASE_URL']);

  useEffect(() => {
    keycloak
      .init({
        onLoad: 'login-required',
        pkceMethod: 'S256',
        redirectUri: process.env['REACT_APP_WEB_BASE_URL'] + '#/sso-login-loading?',
        flow: 'implicit',
      })
      .then(async (auth: any) => {
        if (auth) {
          alert(auth);
        } else {
          alert('Failed to authenticate with the server.');
        }
      })
      .catch((e: any) => {
        alert('Organization not found. Please recheck the organization and try again.');
        localStorage.clear();
      });
  }, []);

  if (!keycloakAuthenticated) {
    return (
      <div className="loader-container">
        <div
          className="loader"
          style={{
            backgroundColor: 'transparent',
            width: '100vw',
            height: '100%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            textAlign: 'center',
            position: 'fixed',
            zIndex: '9999',
          }}
        >
          <MutatingDots
            height="100"
            width="100"
            color="#E74860"
            secondaryColor="#0C3246"
            radius="15"
            ariaLabel="mutating-dots-loading"
            visible={loading}
          />
        </div>
      </div>
    );
  }

  return <p>Loading done {JSON.stringify(keycloakAuthenticated)}</p>;
};

export default LoginView;
