import React, { Component } from 'react';
import { withAuth0 } from '@auth0/auth0-react';


class Profile extends Component {
    
  render() {
    // `this.props.auth0` has all the same properties as the `useAuth0` hook
    const { 
        isLoading,
        isAuthenticated,
        error,
        user} = this.props.auth0;
    if (isLoading){
        return (
            <div class="github">
              <span>Loading...</span>
            </div>
        );
    }
    if (error) {
        return (
            <div class="github">
              <span>There was an error...</span>
            </div>
        );
    }
    if (isAuthenticated) {
        return (
            <span>Hello {user.name}</span>
        );
    } else {
        return <div></div>;
    }    
  }
}

export default withAuth0(Profile);
