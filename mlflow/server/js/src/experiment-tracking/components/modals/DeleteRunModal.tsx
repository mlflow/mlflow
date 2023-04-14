import React, { Component } from 'react';
import { ConfirmModal } from './ConfirmModal';
import { deleteRunApi, openErrorModal } from '../../actions';
import { connect } from 'react-redux';
import Utils from '../../../common/utils/Utils';

type Props = {
  isOpen: boolean;
  onClose: (...args: any[]) => any;
  selectedRunIds: string[];
  openErrorModal: (...args: any[]) => any;
  deleteRunApi: (...args: any[]) => any;
};

export class DeleteRunModalImpl extends Component<Props> {
  constructor(props: Props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleSubmit() {
    const deletePromises: any = [];
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
            {/* @ts-expect-error TS(4111): Property 'SHOW_GDPR_PURGING_MESSAGES' comes from a... Remove this comment to see the full error message */}
            {process.env.SHOW_GDPR_PURGING_MESSAGES === 'true' ? (
              <p>
                Deleted runs are restorable for 30 days, after which they are purged along with
                associated metrics, params, tags, and artifacts.
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
