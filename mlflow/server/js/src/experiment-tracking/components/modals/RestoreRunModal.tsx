/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { ConfirmModal } from './ConfirmModal';
import { connect } from 'react-redux';
import { openErrorModal, restoreRunApi } from '../../actions';
import Utils from '../../../common/utils/Utils';

type Props = {
  isOpen: boolean;
  onClose: (...args: any[]) => any;
  selectedRunIds: string[];
  openErrorModal: (...args: any[]) => any;
  restoreRunApi: (...args: any[]) => any;
  onSuccess?: () => void;
};

export class RestoreRunModalImpl extends Component<Props> {
  constructor(props: Props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleSubmit() {
    const restorePromises: any = [];
    this.props.selectedRunIds.forEach((runId) => {
      restorePromises.push(this.props.restoreRunApi(runId));
    });
    return Promise.all(restorePromises)
      .catch((e) => {
        let errorMessage = 'While restoring an experiment run, an error occurred.';
        if (e.textJson && e.textJson.error_code === 'RESOURCE_LIMIT_EXCEEDED') {
          errorMessage = errorMessage + ' ' + e.textJson.message;
        }
        this.props.openErrorModal(errorMessage);
      })
      .then(() => {
        this.props.onSuccess?.();
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
        confirmButtonText="Restore"
      />
    );
  }
}

const mapDispatchToProps = {
  restoreRunApi,
  openErrorModal,
};

export default connect(null, mapDispatchToProps)(RestoreRunModalImpl);
