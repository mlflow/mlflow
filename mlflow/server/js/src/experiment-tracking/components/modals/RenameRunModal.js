import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { injectIntl } from 'react-intl';

import { GenericInputModal } from './GenericInputModal';
import { RenameForm, NEW_NAME_FIELD } from './RenameForm';

import { updateRunApi } from '../../actions';
import { getUUID } from '../../../common/utils/ActionUtils';

export class RenameRunModalImpl extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    runUuid: PropTypes.string.isRequired,
    runName: PropTypes.string.isRequired,
    onClose: PropTypes.func.isRequired,
    updateRunApi: PropTypes.func.isRequired,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
    onSuccess: PropTypes.func,
  };

  handleRenameRun = (values) => {
    // get value of input field
    const newRunName = values[NEW_NAME_FIELD];

    const updateRunRequestId = getUUID();

    return this.props
      .updateRunApi(this.props.runUuid, newRunName, updateRunRequestId)
      .then(() => this.props.onSuccess?.());
  };

  render() {
    const { isOpen, runName } = this.props;
    return (
      <GenericInputModal
        title={this.props.intl.formatMessage({
          defaultMessage: 'Rename Run',
          description: 'Modal title to rename the experiment run name',
        })}
        okText={this.props.intl.formatMessage({
          defaultMessage: 'Save',
          description: 'Modal button text to save the changes to rename the experiment run name',
        })}
        isOpen={isOpen}
        handleSubmit={this.handleRenameRun}
        onClose={this.props.onClose}
      >
        <RenameForm
          type='run'
          name={runName}
          visible={isOpen}
          validator={async (_, value) => {
            if (typeof value === 'string' && value.length && !value.trim()) {
              throw new Error(
                this.props.intl.formatMessage({
                  defaultMessage: 'Run name cannot consist only of whitespace!',
                  description:
                    "An error shown when user sets the run's name to whitespace characters only",
                }),
              );
            }
            return true;
          }}
        />
      </GenericInputModal>
    );
  }
}

const mapDispatchToProps = {
  updateRunApi,
};

export const RenameRunModalWithIntl = injectIntl(RenameRunModalImpl);
export const RenameRunModal = connect(undefined, mapDispatchToProps)(RenameRunModalWithIntl);
