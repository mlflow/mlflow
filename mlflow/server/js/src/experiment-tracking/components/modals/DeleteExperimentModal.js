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
    activeExperimentIds: PropTypes.arrayOf(PropTypes.string),
    experimentId: PropTypes.string.isRequired,
    experimentName: PropTypes.string.isRequired,
    deleteExperimentApi: PropTypes.func.isRequired,
    listExperimentsApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
  };

  handleSubmit = () => {
    const { experimentId, activeExperimentIds } = this.props;
    const deleteExperimentRequestId = getUUID();

    const deletePromise = this.props
      .deleteExperimentApi(experimentId, deleteExperimentRequestId)
      .then(() => {
        // reload the page if an active experiment was deleted
        if (activeExperimentIds?.includes(experimentId)) {
          if (activeExperimentIds.length === 1) {
            // send it to root
            this.props.history.push(Routes.rootRoute);
          } else {
            const experimentIds = activeExperimentIds.filter((eid) => eid !== experimentId);
            const route =
              experimentIds.length === 1
                ? Routes.getExperimentPageRoute(experimentIds[0])
                : Routes.getCompareExperimentsPageRoute(experimentIds);
            this.props.history.push(route);
          }
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
                Deleted experiments are restorable for 30 days, after which they are purged along
                with their associated runs, including metrics, params, tags, and artifacts.
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
