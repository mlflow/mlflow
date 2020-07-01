import React, { Component } from 'react';
import { ConfirmModal } from './ConfirmModal';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { openErrorModal, restoreRunApi } from '../../actions';
import Utils from '../../../common/utils/Utils';

export class RestoreRunModalImpl extends Component {
  constructor(props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  static propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    selectedRunIds: PropTypes.arrayOf(PropTypes.string).isRequired,
    openErrorModal: PropTypes.func.isRequired,
    restoreRunApi: PropTypes.func.isRequired,
  };

  handleSubmit() {
    const restorePromises = [];
    this.props.selectedRunIds.forEach((runId) => {
      restorePromises.push(this.props.restoreRunApi(runId));
    });
    return Promise.all(restorePromises).catch(() => {
      this.props.openErrorModal('While restoring an experiment run, an error occurred.');
    });
  }

  render() {
    const number = this.props.selectedRunIds.length;
    return (
      <ConfirmModal
        isOpen={this.props.isOpen}
        onClose={this.props.onClose}
        handleSubmit={this.handleSubmit}
        title={`Restore Experiment ${Utils.pluralize('Run', number)}`}
        helpText={`${number} experiment ${Utils.pluralize('run', number)} will be restored.`}
        confirmButtonText={'Restore'}
      />
    );
  }
}

const mapDispatchToProps = {
  restoreRunApi,
  openErrorModal,
};

export default connect(null, mapDispatchToProps)(RestoreRunModalImpl);
