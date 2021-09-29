import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const LoginButton = () => {
  const { loginWithRedirect } = useAuth0();
  return (
    <div className='github'>
      <button className='btn btn-primary' onClick={() => loginWithRedirect()}>
        <span>Log In</span>
      </button>
    </div>
  );
};

export default LoginButton;
