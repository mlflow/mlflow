import React, { Component } from 'react';
import Colors from '../styles/Colors';
import noExperiments from '../../common/static/no-experiments.svg';

const MLFLOW_EXPERIMENT_DOC_URL = 'https://www.mlflow.org/docs/latest/cli.html#experiments';
export class NoExperimentView extends Component {
  render() {
    const mlflowExperimentLink = <a href={MLFLOW_EXPERIMENT_DOC_URL}>mlflow experiments</a>;
    return (
      <div>
        <img
          className='center'
          alt='No experiments found.'
          style={{ height: '200px', marginTop: '80px' }}
          src={noExperiments}
        />
        <h1 className='center' style={{ paddingTop: '10px' }}>
          No Experiments Exist
        </h1>
        <h2 className='center' style={{ color: Colors.secondaryText }}>
          To create an experiment use the {mlflowExperimentLink} CLI.
        </h2>
      </div>
    );
  }
}
