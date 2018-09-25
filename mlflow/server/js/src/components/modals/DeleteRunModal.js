import React, { Component } from 'react';
import { ConfirmModal } from './ConfirmModal';
import PropTypes from 'prop-types';
import { deleteRunApi, openErrorModal } from '../../Actions';
import { connect } from 'react-redux';
import Utils from '../../utils/Utils';

class DeleteRunModal extends Component {
  constructor(props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  static propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    selectedRunIds: PropTypes.arrayOf(String).isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  handleSubmit() {
    const deletePromises = [];
    this.props.selectedRunIds.forEach((runId) => {
      deletePromises.push(this.props.dispatch(deleteRunApi(runId)));
    });
    return Promise.all(deletePromises).catch(() => {
      this.props.dispatch(openErrorModal('While deleting an experiment run, an error occurred.'));
    });
  }

  render() {
    const number = this.props.selectedRunIds.length;
    return (
      <ConfirmModal
        isOpen={this.props.isOpen}
        onClose={this.props.onClose}
        handleSubmit={this.handleSubmit}
        title={`Delete Experiment ${Utils.pluralize("Run", number)}`}
        helpText={`${number} experiment ${Utils.pluralize('run', number)} will be deleted.`}
        confirmButtonText={"Delete"}
      />
    );
  }
}

export default connect()(DeleteRunModal);
