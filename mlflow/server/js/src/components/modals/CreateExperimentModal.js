import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';

import GenericInputModal from './GenericInputModal';
import { CreateExperimentForm, EXP_NAME_FIELD } from './CreateExperimentForm';

import { createExperimentApi } from '../../Actions';

export class CreateExperimentModal extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    onClose: PropTypes.func.isRequired,
    createExperimentApi: PropTypes.func.isRequired,
  };

  handleCreateExperiment = (values) => {
    // get value of input field
    const experimentName = values[EXP_NAME_FIELD];
    return this.props.createExperimentApi(experimentName);
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
};

export default connect(undefined, mapDispatchToProps)(CreateExperimentModal);

