import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';
import debounce from 'lodash/debounce';

import Routes from '../../Routes';
import { GenericInputModal } from './GenericInputModal';
import { CreateExperimentForm, EXP_NAME_FIELD, ARTIFACT_LOCATION } from './CreateExperimentForm';
import { getExperimentNameValidator } from './validation';

import { createExperimentApi, listExperimentsApi, getUUID } from '../../Actions';
import { getExperiments } from '../../reducers/Reducers';

class CreateExperimentModalImpl extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    onClose: PropTypes.func.isRequired,
    experimentNames: PropTypes.arrayOf(String).isRequired,
    createExperimentApi: PropTypes.func.isRequired,
    listExperimentsApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
  };

  handleCreateExperiment = (values) => {
    // get values of input fields
    const experimentName = values[EXP_NAME_FIELD];
    const artifactLocation = values[ARTIFACT_LOCATION];

    // The listExperimentsPromise needs to be fulfilled before redirecting the user
    // to the newly created experiment page (history.push()).
    // At the same time, the result value of the createExperimentPromise is needed
    // to get the experiment id. Thus, the state has to be shared through the promise chain.
    const createExperimentPromise = this.props
      .createExperimentApi(experimentName, artifactLocation, getUUID())
      .then(({ value }) => {
        const listExperimentsPromise = this.props.listExperimentsApi(getUUID());
        return Promise.all([value, listExperimentsPromise]);
      })
      .then(([value, _]) => {
        if (value && value.experiment_id) {
          // redirect user to newly created experiment page
          this.props.history.push(Routes.getExperimentPageRoute(value.experiment_id));
        }
      });

    return createExperimentPromise;
  };

  debouncedExperimentNameValidator = debounce(
    getExperimentNameValidator(() => this.props.experimentNames),
    400,
  );

  render() {
    const { isOpen } = this.props;

    const inputComponent = (
      <CreateExperimentForm visible={isOpen} validator={this.debouncedExperimentNameValidator} />
    );

    return (
      <GenericInputModal
        title='Create Experiment'
        childForm={inputComponent}
        isOpen={isOpen}
        handleSubmit={this.handleCreateExperiment}
        onClose={this.props.onClose}
      />
    );
  }
}

const mapStateToProps = (state) => {
  const experiments = getExperiments(state);
  const experimentNames = experiments.map((e) => e.getName());
  return { experimentNames };
};

const mapDispatchToProps = {
  createExperimentApi,
  listExperimentsApi,
};

export const CreateExperimentModal = withRouter(
  connect(mapStateToProps, mapDispatchToProps)(CreateExperimentModalImpl)
);
