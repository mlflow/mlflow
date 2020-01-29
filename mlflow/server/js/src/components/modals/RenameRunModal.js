import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Modal } from 'antd';

import { RenameFormView, NEW_NAME_FIELD } from './RenameFormView';
import { setTagApi, getUUID, openErrorModal } from '../../Actions';
import Utils from '../../utils/Utils';

export class RenameRunModal extends Component {
  state = {
    isSubmittingState: false,
  };

  static propTypes = {
    isOpen: PropTypes.bool,
    experimentId: PropTypes.number.isRequired,
    runUuid: PropTypes.string.isRequired,
    runName: PropTypes.string.isRequired,
    onClose: PropTypes.func.isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  handleRenameRun = () => {
    this.form.validateFields((err, values) => {
      if (!err) {
        this.setState({ isSubmittingState: true });
        const newRunName = values[NEW_NAME_FIELD];
        const tagKey = Utils.runNameTag;
        const setTagRequestId = getUUID();

        this.props.dispatch(setTagApi(this.props.runUuid, tagKey, newRunName, setTagRequestId))
          .then(this.resetAndClearModalForm)
          .then(this.onRequestCloseHandler)
          .then(this.handleSubmitFailure);
      }
    });
  }

  resetAndClearModalForm = () => {
    this.setState({ isSubmittingState: false });
    this.form.resetFields();
  };

  handleRegistrationFailure = (e) => {
    this.setState({ isSubmittingState: false });
    Utils.logErrorAndNotifyUser(e);
    this.props.dispatch(openErrorModal('While renaming a run, an error occurred.'));
  };

  onRequestCloseHandler = () => {
    if (!this.state.isSubmittingState) {
      this.resetAndClearModalForm();
      this.props.onClose();
    }
  }

  saveFormRef = (form) => {
    this.form = form;
  };

  saveFormComponentRef = (formComponent) => {
    this.formComponent = formComponent;
  };

  render() {
    const { isSubmittingState } = this.state;
    const { isOpen, runName } = this.props;

    return (
      <Modal
        title='Rename Run'
        width={540}
        visible={isOpen}
        onOk={this.handleRenameRun}
        okText='Save'
        confirmLoading={isSubmittingState}
        onCancel={this.onRequestCloseHandler}
        centered
      >
        <RenameFormView
          type='run'
          name={runName}
          visible={isOpen}
          ref={this.saveFormRef}
          wrappedComponentRef={this.saveFormComponentRef}
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

export default connect(null, mapDispatchToProps)(RenameRunModal);
