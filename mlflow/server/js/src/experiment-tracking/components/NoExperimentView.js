import React, { Component } from 'react';
import Colors from '../styles/Colors';
import noExperiments from '../../common/static/no-experiments.svg';

export class NoExperimentView extends Component {
  render() {
    return (
      <div>
        <img
          className='center'
          alt="No experiments found."
          style={{ height: '200px', marginTop: '80px' }}
          src={noExperiments}
        />
        <h1 className="center" style={{ paddingTop: '10px' }}>
          No Experiments Exist
        </h1>
        <h2 className="center" style={{ color: Colors.secondaryText }}>
          To create an experiment use the{' '}
          <a href={"https://www.mlflow.org/docs/latest/cli.html#experiments"}>
            mlflow experiments
          </a>{' '}
          CLI.
        </h2>
      </div>
    );
  }
}
