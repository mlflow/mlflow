import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Modal } from 'antd';

import Utils from '../../../common/utils/Utils';

/**
 * Generic modal that has a title and an input field with a save/submit button.
 * As of now, it is used to display the 'Rename Run' and 'Rename Experiment' modals.
 */
export class GenericInputModal extends Component {
  state = {
    isSubmitting: false,
  };

  static propTypes = {
    okText: PropTypes.string,
    cancelText: PropTypes.string,
    isOpen: PropTypes.bool,
    onClose: PropTypes.func.isRequired,
    onCancel: PropTypes.func,
    className: PropTypes.string,
    footer: PropTypes.node,
    // Function which returns a promise which resolves when the submission is done.
    handleSubmit: PropTypes.func.isRequired,
    title: PropTypes.node.isRequired,
    // Antd Form
    children: PropTypes.node.isRequired,
  };

  onSubmit = () => {
    return this.form.validateFields((err, values) => {
      if (!err) {
        this.setState({ isSubmitting: true });

        // call handleSubmit from parent component, pass form values
        // handleSubmit is expected to return a promise
        return this.props
          .handleSubmit(values)
          .then(this.resetAndClearModalForm)
          .catch(this.handleSubmitFailure)
          .finally(this.onRequestCloseHandler);
      }
      return Promise.reject(err);
    });
  };

  resetAndClearModalForm = () => {
    this.setState({ isSubmitting: false });
    this.form.resetFields();
  };

  handleSubmitFailure = (e) => {
    this.setState({ isSubmitting: false });
    Utils.logErrorAndNotifyUser(e);
  };

  onRequestCloseHandler = () => {
    if (!this.state.isSubmitting) {
      this.resetAndClearModalForm();
      this.props.onClose();
    }
  };

  handleCancel = () => {
    this.onRequestCloseHandler();
    // Check for optional `onCancel` method prop.
    if (this.props.onCancel !== undefined) {
      this.props.onCancel();
    }
  };

  saveFormRef = (form) => {
    this.form = form;
  };

  render() {
    const { isSubmitting } = this.state;
    const { okText, cancelText, isOpen, footer, children } = this.props;

    // add props (ref) to passed component
    const displayForm = React.cloneElement(children, { ref: this.saveFormRef });

    return (
      <Modal
        className={this.props.className}
        title={this.props.title}
        width={540}
        visible={isOpen}
        onOk={this.onSubmit}
        okText={okText}
        cancelText={cancelText}
        confirmLoading={isSubmitting}
        onCancel={this.handleCancel}
        footer={footer}
        centered
      >
        {displayForm}
      </Modal>
    );
  }
}
