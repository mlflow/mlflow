import React, { Component } from 'react';
import { ConfirmModal } from './ConfirmModal';
import PropTypes from 'prop-types';
import { deleteRunApi, openErrorModal } from '../../actions';
import { connect } from 'react-redux';
import Utils from '../../../common/utils/Utils';

export class DeleteRunModalImpl extends Component {
  constructor(props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  static propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    selectedRunIds: PropTypes.arrayOf(PropTypes.string).isRequired,
    openErrorModal: PropTypes.func.isRequired,
    deleteRunApi: PropTypes.func.isRequired,
  };

  handleSubmit() {
    const deletePromises = [];
    this.props.selectedRunIds.forEach((runId) => {
      deletePromises.push(this.props.deleteRunApi(runId));
    });
    return Promise.all(deletePromises).catch(() => {
      this.props.openErrorModal('While deleting an experiment run, an error occurred.');
    });
  }

  render() {
    const number = this.props.selectedRunIds.length;
    return (
      <ConfirmModal
        isOpen={this.props.isOpen}
        onClose={this.props.onClose}
        handleSubmit={this.handleSubmit}
        title={`Delete Experiment ${Utils.pluralize('Run', number)}`}
        helpText={
          <div>
            <p>
              <b>
                {number} experiment {Utils.pluralize('run', number)} will be deleted.
              </b>
            </p>
            {process.env.SHOW_GDPR_PURGING_MESSAGES === 'true' ? (
              <p>
                Deleted runs are restorable for 30 days, after which they are purged along with
                associated metrics, params and tags.
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
  deleteRunApi,
  openErrorModal,
};

export default connect(null, mapDispatchToProps)(DeleteRunModalImpl);
