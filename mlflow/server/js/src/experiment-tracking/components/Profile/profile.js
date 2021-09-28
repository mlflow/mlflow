import React from 'react';
import { withAuthenticationRequired, useAuth0 } from '@auth0/auth0-react';

const Profile = () => {
  const { user } = useAuth0();
  if(user.name) {
    return <span>Hello {user.name}</span>
  } 
  else {
    return <span></span>
  }
};

export default withAuthenticationRequired(Profile);