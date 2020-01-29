import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Modal } from 'antd';

import { RenameFormView, NEW_NAME_FIELD } from './RenameFormView';
import { updateExperimentApi, openErrorModal } from '../../Actions';
import Utils from '../../utils/Utils';


export class RenameExperimentModal extends Component {
  state = {
    isSubmittingState: false,
  };

  static propTypes = {
    isOpen: PropTypes.bool,
    experimentId: PropTypes.number,
    experimentName: PropTypes.string,
    onClose: PropTypes.func.isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  handleRenameExperiment = () => {
    this.form.validateFields((err, values) => {
      if (!err) {
        this.setState({ isSubmittingState: true });
        const newExperimentName = values[NEW_NAME_FIELD];

        this.props.dispatch(updateExperimentApi(this.props.experimentId, newExperimentName))
        .then(this.resetAndClearModalForm)
        .then(this.onRequestCloseHandler)
        .catch(this.handleSubmitFailure);
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
    this.props.dispatch(openErrorModal('While renaming an experiment, an error occurred.'));
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
    const { isOpen, experimentName } = this.props;
    return (
      <Modal
        title='Rename Experiment'
        width={540}
        visible={isOpen}
        onOk={this.handleRenameExperiment}
        okText='Save'
        confirmLoading={isSubmittingState}
        onCancel={this.onRequestCloseHandler}
        centered
      >
        <RenameFormView
          type='experiment'
          name={experimentName}
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

export default connect(null, mapDispatchToProps)(RenameExperimentModal);

