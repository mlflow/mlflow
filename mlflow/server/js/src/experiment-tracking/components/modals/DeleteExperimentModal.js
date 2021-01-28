import React, { Component } from 'react';
import { ConfirmModal } from './ConfirmModal';
import PropTypes from 'prop-types';
import { deleteExperimentApi, listExperimentsApi } from '../../actions';
import Routes from '../../routes';
import Utils from '../../../common/utils/Utils';
import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';
import { getUUID } from '../../../common/utils/ActionUtils';

export class DeleteExperimentModalImpl extends Component {
  static propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    activeExperimentId: PropTypes.string,
    experimentId: PropTypes.string.isRequired,
    experimentName: PropTypes.string.isRequired,
    deleteExperimentApi: PropTypes.func.isRequired,
    listExperimentsApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
  };

  handleSubmit = () => {
    const { experimentId, activeExperimentId } = this.props;
    const deleteExperimentRequestId = getUUID();

    const deletePromise = this.props
      .deleteExperimentApi(experimentId, deleteExperimentRequestId)
      .then(() => {
        // check whether the deleted experiment is currently selected
        if (experimentId === activeExperimentId) {
          // navigate to root URL and let route pick the next active experiment to show
          this.props.history.push(Routes.rootRoute);
        }
      })
      .then(() => this.props.listExperimentsApi(deleteExperimentRequestId))
      .catch((e) => {
        Utils.logErrorAndNotifyUser(e);
      });

    return deletePromise;
  };

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
                Experiment "{this.props.experimentName}" (Experiment ID: {this.props.experimentId})
                will be deleted.
              </b>
            </p>
            {process.env.SHOW_GDPR_PURGING_MESSAGES === 'true' ? (
              <p>
                Deleted experiments are restorable for 30 days, after which they are purged.
                <br />
                Artifacts are not automatically purged and must be manually deleted.
              </p>
            ) : (
              ''
            )}
          </div>
        }
        confirmButtonText={'Delete'}
      />
    );
  }
}

const mapDispatchToProps = {
  deleteExperimentApi,
  listExperimentsApi,
};

export const DeleteExperimentModal = withRouter(
  connect(undefined, mapDispatchToProps)(DeleteExperimentModalImpl),
);
