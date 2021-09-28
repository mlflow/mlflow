import React from "react";
import { useAuth0 } from "@auth0/auth0-react";

const LoginButton = () => {
  const { loginWithRedirect } = useAuth0();
  return (
    <a onClick={() => loginWithRedirect()}>
      <div className='github'>
        <span>Log In</span>
      </div>
    </a>
  );
};

export default LoginButton;
