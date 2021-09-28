import React from 'react';
import AuthenticationButton from './authentication-button';
import Profile from '../Profile/profile';

const AuthNav = () => (
  <div className='auth-nav'>
    <AuthenticationButton />
    <Profile />
  </div>
);

export default AuthNav;
