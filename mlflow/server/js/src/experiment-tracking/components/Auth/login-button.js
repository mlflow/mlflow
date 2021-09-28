import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const LoginButton = () => {
  const { loginWithRedirect } = useAuth0();
  return (
    <button onClick={() => loginWithRedirect()}>
      <div className='github'>
        <span>Log In</span>
      </div>
    </button>
  );
};

export default LoginButton;
