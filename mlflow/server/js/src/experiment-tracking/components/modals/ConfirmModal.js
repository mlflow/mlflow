import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Modal } from 'antd';

export class ConfirmModal extends Component {
  constructor(props) {
    super(props);
    this.onRequestCloseHandler = this.onRequestCloseHandler.bind(this);
    this.handleSubmitWrapper = this.handleSubmitWrapper.bind(this);
  }

  static propTypes = {
    isOpen: PropTypes.bool.isRequired,
    // Function which returns a promise which resolves when the submission is done.
    handleSubmit: PropTypes.func.isRequired,
    onClose: PropTypes.func.isRequired,
    title: PropTypes.string.isRequired,
    helpText: PropTypes.node.isRequired,
    confirmButtonText: PropTypes.string.isRequired,
  };

  state = {
    isSubmitting: false,
  };

  onRequestCloseHandler() {
    if (!this.state.isSubmitting) {
      this.props.onClose();
    }
  }

  handleSubmitWrapper() {
    this.setState({ isSubmitting: true });
    return this.props.handleSubmit().finally(() => {
      this.props.onClose();
      this.setState({ isSubmitting: false });
    });
  }

  render() {
    return (
      <Modal
        title={this.props.title}
        visible={this.props.isOpen}
        onOk={this.handleSubmitWrapper}
        okText={this.props.confirmButtonText}
        confirmLoading={this.state.isSubmitting}
        onCancel={this.onRequestCloseHandler}
        centered
      >
        <div className='modal-explanatory-text'>{this.props.helpText}</div>
      </Modal>
    );
  }
}
