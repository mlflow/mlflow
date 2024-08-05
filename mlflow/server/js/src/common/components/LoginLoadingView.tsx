import React, { useEffect, useState } from 'react';
import keycloak from '../../keycloak-config';
import { useNavigate } from '../utils/RoutingUtils';
import { MutatingDots } from 'react-loader-spinner';

const LoginLoadingView = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);

  function keycloakSetup(realm: string) {
    if (typeof window !== 'undefined') {
      keycloak
        .init({
          onLoad: 'check-sso',
          pkceMethod: 'S256',
          redirectUri: process.env['REACT_APP_WEB_BASE_URL'] + '#/sso-login-loading?',
          flow: 'implicit',
        })
        .then(async (auth: any) => {
          if (auth) {
            localStorage.setItem('authToken', keycloak?.token);
            const path = localStorage.getItem('loginPath');
            window.location.href = '' + path;
          } else {
            alert('Error: Authentication failed');
          }
        })
        .catch((e: any) => {
          console.error('Error initializing Keycloak:', e);
          alert('Error: Unable to initialize Keycloak');
        });
    }
  }

  useEffect(() => {
    keycloakSetup('nuodata');
  }, []);

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
};

export default LoginLoadingView;
