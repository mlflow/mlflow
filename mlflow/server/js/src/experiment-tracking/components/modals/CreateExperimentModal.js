import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';
import debounce from 'lodash/debounce';

import Routes from '../../routes';
import { GenericInputModal } from './GenericInputModal';
import { CreateExperimentForm, EXP_NAME_FIELD, ARTIFACT_LOCATION } from './CreateExperimentForm';
import { getExperimentNameValidator } from '../../../common/forms/validations';

import { createExperimentApi, listExperimentsApi } from '../../actions';
import { getExperiments } from '../../reducers/Reducers';

export class CreateExperimentModalImpl extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    onClose: PropTypes.func.isRequired,
    experimentNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    createExperimentApi: PropTypes.func.isRequired,
    listExperimentsApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
  };

  handleCreateExperiment = async (values) => {
    // get values of input fields
    const experimentName = values[EXP_NAME_FIELD];
    const artifactLocation = values[ARTIFACT_LOCATION];

    // Both createExperimentApi and listExperimentsApi calls need to be fulfilled sequentially
    // before redirecting the user to the newly created experiment page (history.push())
    const response = await this.props.createExperimentApi(experimentName, artifactLocation);
    await this.props.listExperimentsApi();

    const {
      value: { experiment_id: newExperimentId },
    } = response;
    if (newExperimentId) {
      this.props.history.push(Routes.getExperimentPageRoute(newExperimentId));
    }
  };

  debouncedExperimentNameValidator = debounce(
    getExperimentNameValidator(() => this.props.experimentNames),
    400,
  );

  render() {
    const { isOpen } = this.props;
    return (
      <GenericInputModal
        title='Create Experiment'
        okText='Create'
        isOpen={isOpen}
        handleSubmit={this.handleCreateExperiment}
        onClose={this.props.onClose}
      >
        <CreateExperimentForm visible={isOpen} validator={this.debouncedExperimentNameValidator} />
      </GenericInputModal>
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
  connect(mapStateToProps, mapDispatchToProps)(CreateExperimentModalImpl),
);
