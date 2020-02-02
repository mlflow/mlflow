import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';

import GenericInputModal from './GenericInputModal';
import { RenameForm, NEW_NAME_FIELD } from './RenameForm';

import { updateExperimentApi } from '../../Actions';

export class RenameExperimentModal extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    experimentId: PropTypes.number,
    experimentName: PropTypes.string,
    onClose: PropTypes.func.isRequired,
    updateExperimentApi: PropTypes.func.isRequired,
  };

  handleRenameExperiment = (values) => {
    // get value of input field
    const newExperimentName = values[NEW_NAME_FIELD];
    return this.props.updateExperimentApi(this.props.experimentId, newExperimentName);
  }

  render() {
    const { isOpen, experimentName } = this.props;

    const inputComponent = <RenameForm
      type='experiment'
      name={experimentName}
      visible={isOpen}
    />;

    return (
      <GenericInputModal
        title='Rename Experiment'
        childForm={inputComponent}
        isOpen={isOpen}
        handleSubmit={this.handleRenameExperiment}
        onClose={this.props.onClose}
        errorMessage='While renaming an experiment, an error occurred.'
      />
    );
  }
}

const mapDispatchToProps = {
  updateExperimentApi,
};

export default connect(undefined, mapDispatchToProps)(RenameExperimentModal);

