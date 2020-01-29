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
    dispatch: PropTypes.func.isRequired,
  };

  handleRenameExperiment = (newExperimentName) => {
    return this.props.dispatch(updateExperimentApi(this.props.experimentId, newExperimentName));
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
        errorMessage='While renaming a run, an error occurred.'
      />
    );
  }
}

// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
  };
};

export default connect(null, mapDispatchToProps)(RenameExperimentModal);

