import React, { Component } from 'react';
import PropTypes from 'prop-types';
import spinner from '../static/mlflow-spinner.png';
import classNames from 'classnames';
import './Spinner.css';

export class Spinner extends Component {
  static propTypes = {
    showImmediately: PropTypes.bool,
  };

  render() {
    const className = classNames({
      Spinner: true,
      'Spinner-immediate': this.props.showImmediately,
    });
    return (
      <div className={className}>
        <img alt='Page loading...' src={spinner} />
      </div>
    );
  }
}
