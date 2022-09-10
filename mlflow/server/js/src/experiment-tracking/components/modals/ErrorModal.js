import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Modal } from 'antd';
import { connect } from 'react-redux';
import { getErrorModalText, isErrorModalOpen } from '../../reducers/Reducers';
import { closeErrorModal } from '../../actions';
import { injectIntl } from 'react-intl';

export class ErrorModalImpl extends Component {
  static propTypes = {
    isOpen: PropTypes.bool.isRequired,
    onClose: PropTypes.func.isRequired,
    text: PropTypes.string.isRequired,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

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
        centered
      >
        <p style={{ marginBottom: '10px' }}>{this.props.text}</p>
      </Modal>
    );
  }
}

const mapStateToProps = (state) => {
  const isOpen = isErrorModalOpen(state);
  const text = getErrorModalText(state);
  return {
    isOpen,
    text,
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    onClose: () => {
      dispatch(closeErrorModal());
    },
  };
};

export const ErrorModalWithIntl = injectIntl(ErrorModalImpl);
export default connect(mapStateToProps, mapDispatchToProps)(ErrorModalWithIntl);
