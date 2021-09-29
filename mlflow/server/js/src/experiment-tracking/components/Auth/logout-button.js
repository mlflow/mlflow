import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const LogoutButton = () => {
  const { logout } = useAuth0();
  return (
    <div className='github'>
      <button
        className='btn btn-primary'
        onClick={() =>
          logout({
            returnTo: window.location.origin,
          })
        }
      >
        <span>Log Out</span>
      </button>
    </div>
  );
};

export default LogoutButton;
