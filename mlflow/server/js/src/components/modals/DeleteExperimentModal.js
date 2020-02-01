import React, { Component } from 'react';
import { ConfirmModal } from './ConfirmModal';
import PropTypes from 'prop-types';
import { deleteExperimentApi, openErrorModal } from '../../Actions';
import { connect } from 'react-redux';

class DeleteExperimentModal extends Component {
  static propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    experimentId: PropTypes.number.isRequired,
    experimentName: PropTypes.string.isRequired,
    deleteExperimentApi: PropTypes.func.isRequired,
    openErrorModal: PropTypes.func.isRequired,
  };

  handleSubmit = () => {
    const deletePromise = this.props.deleteExperimentApi(this.props.experimentId).catch(() => {
      this.props.openErrorModal('While deleting an experiment, an error occurred.');
    });

    return deletePromise;
  }

  render() {
    return (
      <ConfirmModal
        isOpen={this.props.isOpen}
        onClose={this.props.onClose}
        handleSubmit={this.handleSubmit}
        title={`Delete Experiment "${this.props.experimentName}"`}
        helpText={
          <div>
            <p>
              <b>
                Experiment "{this.props.experimentName}"
                (Experiment ID: {this.props.experimentId}) will be deleted.
              </b>
            </p>
            {
              process.env.SHOW_GDPR_PURGING_MESSAGES === 'true' ?
              <p>
                Deleted experiments are restorable for 30 days, after which they are purged.
                <br />
                Artifacts are not automatically purged and must be manually deleted.
              </p> : ""
            }
          </div>
        }
        confirmButtonText={"Delete"}
      />
    );
  }
}

const mapDispatchToProps = {
  deleteExperimentApi,
  openErrorModal,
};

export default connect(undefined, mapDispatchToProps)(DeleteExperimentModal);
