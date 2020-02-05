import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';

import Routes from '../../Routes';
import GenericInputModal from './GenericInputModal';
import { CreateExperimentForm, EXP_NAME_FIELD } from './CreateExperimentForm';

import { createExperimentApi, listExperimentsApi, getUUID } from '../../Actions';

export class CreateExperimentModal extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    onClose: PropTypes.func.isRequired,
    createExperimentApi: PropTypes.func.isRequired,
    listExperimentsApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
  };

  handleCreateExperiment = (values) => {
    // get value of input field
    const experimentName = values[EXP_NAME_FIELD];
    const createRequestId = getUUID();
    const createExperimentPromise = this.props.createExperimentApi(experimentName, createRequestId);
    const listExperimentsPromise = this.props.listExperimentsApi(createRequestId);

    // The listExperimentsPromise needs to be fulfilled before redirecting the user
    // to the newly created experiment page (history.push()).
    // At the same time, the result value of the createExperimentPromise is needed
    // to get the experiment id. Thus, the flat promise chain is broken up.
    const returnPromise = Promise.all([createExperimentPromise, listExperimentsPromise])
      .then(([{ value }, _]) => {
        if (value && value.experiment_id) {
          // redirect user to newly created experiment page
          this.props.history.push(Routes.getExperimentPageRoute(value.experiment_id));
        }
      });

    return returnPromise;
  }

  render() {
    const { isOpen } = this.props;

    const inputComponent = <CreateExperimentForm
      visible={isOpen}
    />;

    return (
      <GenericInputModal
        title='Create Experiment'
        childForm={inputComponent}
        isOpen={isOpen}
        handleSubmit={this.handleCreateExperiment}
        onClose={this.props.onClose}
        errorMessage='While creating a new experiment, an error occurred.'
      />
    );
  }
}

const mapDispatchToProps = {
  createExperimentApi,
  listExperimentsApi,
};

export default withRouter(connect(undefined, mapDispatchToProps)(CreateExperimentModal));

