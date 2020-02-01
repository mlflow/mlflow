import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';

import GenericInputModal from './GenericInputModal';

import { updateExperimentApi } from '../../Actions';

export class RenameExperimentModal extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    experimentId: PropTypes.number,
    experimentName: PropTypes.string,
    onClose: PropTypes.func.isRequired,
    updateExperimentApi: PropTypes.func.isRequired,
  };

  handleRenameExperiment = (newExperimentName) => {
    return this.props.updateExperimentApi(this.props.experimentId, newExperimentName);
  }

  render() {
    const { isOpen, experimentName } = this.props;

    return (
      <GenericInputModal
        title='Rename Experiment'
        type='experiment'
        isOpen={isOpen}
        defaultValue={experimentName}
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

