import React from 'react';
import { Route, Redirect, RouteProps } from 'react-router-dom';

export const ProtectedRoute: React.FC<RouteProps> = ({ component, ...rest }) => {
  if (localStorage.getItem('access_token') === null) {
    return <Redirect to='/signin' />;
  } else {
    return <Route component={component} {...rest} />;
  }
};
