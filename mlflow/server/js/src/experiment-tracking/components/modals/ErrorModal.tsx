import React, { Component } from 'react';
import { Modal } from '@databricks/design-system';
import { connect } from 'react-redux';
import { getErrorModalText, isErrorModalOpen } from '../../reducers/Reducers';
import { closeErrorModal } from '../../actions';
import { injectIntl } from 'react-intl';

type ErrorModalImplProps = {
  isOpen: boolean;
  onClose: (...args: any[]) => any;
  text: string;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
};

export class ErrorModalImpl extends Component<ErrorModalImplProps> {
  render() {
    return (
      <Modal
        title={this.props.intl.formatMessage({
          defaultMessage: 'Oops!',
          description: 'Error modal title to rendering errors',
        })}
        visible={this.props.isOpen}
        onCancel={this.props.onClose}
        okButtonProps={{
          style: {
            display: 'none',
          },
        }}
        cancelText={this.props.intl.formatMessage({
          defaultMessage: 'Close',
          description: 'Error modal close button text',
        })}
        // @ts-expect-error TS(2322): Type '{ children: Element; title: any; visible: bo... Remove this comment to see the full error message
        centered
      >
        <p style={{ marginBottom: '10px' }}>{this.props.text}</p>
      </Modal>
    );
  }
}

const mapStateToProps = (state: any) => {
  const isOpen = isErrorModalOpen(state);
  const text = getErrorModalText(state);
  return {
    isOpen,
    text,
  };
};

const mapDispatchToProps = (dispatch: any) => {
  return {
    onClose: () => {
      dispatch(closeErrorModal());
    },
  };
};

// @ts-expect-error TS(2769): No overload matches this call.
export const ErrorModalWithIntl = injectIntl(ErrorModalImpl);
export default connect(mapStateToProps, mapDispatchToProps)(ErrorModalWithIntl);
