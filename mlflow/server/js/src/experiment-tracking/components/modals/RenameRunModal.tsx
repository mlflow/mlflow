/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';
import { IntlShape, injectIntl } from 'react-intl';

import { GenericInputModal } from './GenericInputModal';
import { RenameForm, NEW_NAME_FIELD } from './RenameForm';

import { updateRunApi } from '../../actions';
import { getUUID } from '../../../common/utils/ActionUtils';

type Props = {
  isOpen?: boolean;
  runUuid: string;
  runName: string;
  onClose: (...args: any[]) => any;
  updateRunApi: (...args: any[]) => any;
  intl: IntlShape;
  onSuccess?: (...args: any[]) => any;
};

export class RenameRunModalImpl extends Component<Props> {
  handleRenameRun = (values: any) => {
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
          type="run"
          name={runName}
          // @ts-expect-error TS(2769): No overload matches this call.
          visible={isOpen}
          validator={async (_, value) => {
            if (typeof value === 'string' && value.length && !value.trim()) {
              throw new Error(
                this.props.intl.formatMessage({
                  defaultMessage: 'Run name cannot consist only of whitespace!',
                  description: "An error shown when user sets the run's name to whitespace characters only",
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
