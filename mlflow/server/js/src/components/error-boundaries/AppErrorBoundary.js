import React, { Component } from 'react';
import './AppErrorBoundary.css';
import niagara from '../../static/niagara.jpg';

class AppErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  componentDidCatch(error, info) {
    this.setState({ hasError: true });
    console.error(error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div>
          <h1 className={"center"}>Oops! Something went wrong.</h1>
          <h4 className={"center"}>If this error persists, please report an issue to our Github.</h4>
          <img className="niagara" alt="Niagara falls picture." src={niagara}/>
        </div>
      );
    }
    return this.props.children;
  }
}

export default AppErrorBoundary;