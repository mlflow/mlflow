import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Modal } from 'antd';

import { InputFormView, NEW_NAME_FIELD } from './InputFormView';
import { openErrorModal } from '../../Actions';
import Utils from '../../utils/Utils';

/**
 * Generic modal that has a title and an input field with a save/submit button.
 * As of now, it is used to display the 'Rename Run' and 'Rename Experiment' modals.
 */
export class GenericInputModal extends Component {
  state = {
    isSubmitting: false,
  };

  static propTypes = {
    isOpen: PropTypes.bool,
    defaultValue: PropTypes.string.isRequired,
    onClose: PropTypes.func.isRequired,
    // Function which returns a promise which resolves when the submission is done.
    handleSubmit: PropTypes.func.isRequired,
    errorMessage: PropTypes.string.isRequired,
    title: PropTypes.string.isRequired,
    type: PropTypes.string.isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  onSubmit = () => {
    this.form.validateFields((err, values) => {
      if (!err) {
        this.setState({ isSubmitting: true });

        // get value of input field
        const newName = values[NEW_NAME_FIELD];
        // call handleSubmit from parent component, pass input field value
        // handleSubmit is expected to return a promise
        this.props.handleSubmit(newName)
          .then(this.resetAndClearModalForm)
          .catch(this.handleSubmitFailure)
          .finally(this.onRequestCloseHandler);
      }
    });
  }

  resetAndClearModalForm = () => {
    this.setState({ isSubmitting: false });
    this.form.resetFields();
  };

  handleSubmitFailure = (e) => {
    this.setState({ isSubmitting: false });
    Utils.logErrorAndNotifyUser(e);
    this.props.dispatch(openErrorModal(this.props.errorMessage));
  };

  onRequestCloseHandler = () => {
    if (!this.state.isSubmitting) {
      this.resetAndClearModalForm();
      this.props.onClose();
    }
  }

  saveFormRef = (form) => {
    this.form = form;
  };

  render() {
    const { isSubmitting } = this.state;
    const { isOpen, defaultValue } = this.props;

    return (
      <Modal
        title={this.props.title}
        width={540}
        visible={isOpen}
        onOk={this.onSubmit}
        okText='Save'
        confirmLoading={isSubmitting}
        onCancel={this.onRequestCloseHandler}
        centered
      >
        <InputFormView
          type={this.props.type}
          name={defaultValue}
          visible={isOpen}
          ref={this.saveFormRef}
        />
      </Modal>
    );
  }
}

// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
  };
};

export default connect(null, mapDispatchToProps)(GenericInputModal);
