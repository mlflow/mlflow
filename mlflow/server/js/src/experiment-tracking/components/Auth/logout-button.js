import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const LogoutButton = () => {
  const { logout } = useAuth0();
  return (
    <button
      onClick={() =>
        logout({
          returnTo: window.location.origin,
        })
      }
    >
      <div className='github'>
        <span>Log Out</span>
      </div>
    </button>
  );
};

export default LogoutButton;
