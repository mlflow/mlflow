import React from "react";
import { Auth0Provider } from "@auth0/auth0-react";

const Auth0ProviderWithHistory = ({ children }) => {
  const domain = process.env.REACT_APP_AUTH0_DOMAIN;
  const clientId = process.env.REACT_APP_AUTH0_CLIENT_ID;
  const audience = process.env.REACT_APP_AUTH0_AUDIENCE;
  const callbackUrl = process.env.REACT_APP_AUTH0_CALLBACK_URL;
  const useRefreshTokens = process.env.REACT_APP_AUTH0_REFRESH_TOKENS;

  return (
    <Auth0Provider
      domain={domain}
      clientId={clientId}
      audience={audience}
      redirectUri={callbackUrl}
      useRefreshTokens={useRefreshTokens}
    >
      {children}
    </Auth0Provider>
  );
};

export default Auth0ProviderWithHistory;
