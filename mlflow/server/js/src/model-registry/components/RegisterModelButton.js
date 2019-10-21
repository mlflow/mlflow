import React from 'react';
import { Modal, Button } from 'antd';
import {
  RegisterModelForm,
  CREATE_NEW_MODEL_OPTION_VALUE,
  SELECTED_MODEL_FIELD,
  MODEL_NAME_FIELD,
} from './RegisterModelForm';
import { getUUID } from '../../Actions';
import {
  createRegisteredModelApi,
  createModelVersionApi,
  listRegisteredModelsApi,
  searchModelVersionsApi,
} from '../actions';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import Utils from '../../utils/Utils';

class RegisterModelButton extends React.Component {
  static propTypes = {
    disabled: PropTypes.bool.isRequired,
    runUuid: PropTypes.string.isRequired,
    modelPath: PropTypes.string,
    modelByName: PropTypes.object.isRequired,
    createRegisteredModelApi: PropTypes.func.isRequired,
    createModelVersionApi: PropTypes.func.isRequired,
    listRegisteredModelsApi: PropTypes.func.isRequired,
    searchModelVersionsApi: PropTypes.func.isRequired,
  };

  state = {
    visible: false,
    confirmLoading: false,
  };

  createRegisteredModelRequestId = getUUID();

  createModelVersionRequestId = getUUID();

  searchModelVersionRequestId = getUUID();

  showRegisterModal = () => {
    this.setState({ visible: true });
  };

  hideRegisterModal = () => {
    this.setState({ visible: false });
  };

  resetAndClearModalForm = () => {
    this.setState({ visible: false, confirmLoading: false });
    this.form.resetFields();
    this.formComponent.handleModelSelectChange();
  };

  handleRegistrationFailure = (e) => {
    this.setState({ confirmLoading: false });
    Utils.logErrorAndNotifyUser(e);
  };

  reloadModelVersionsForCurrentRun = () => {
    const { runUuid } = this.props;
    return this.props.searchModelVersionsApi({ run_id: runUuid }, this.searchModelVersionRequestId);
  };

  handleRegisterModel = () => {
    this.form.validateFields((err, values) => {
      if (!err) {
        this.setState({ confirmLoading: true });
        const { runUuid, modelPath } = this.props;
        const selectedModelName = values[SELECTED_MODEL_FIELD];
        if (selectedModelName === CREATE_NEW_MODEL_OPTION_VALUE) {
          // When user choose to create a new registered model during the registration, we need to
          // 1. Create a new registered model
          // 2. Create model version #1 in the new registered model
          this.props
            .createRegisteredModelApi(values[MODEL_NAME_FIELD], this.createRegisteredModelRequestId)
            .then(() =>
              this.props.createModelVersionApi(
                values[MODEL_NAME_FIELD],
                modelPath,
                runUuid,
                this.createModelVersionRequestId,
              ),
            )
            .then(this.resetAndClearModalForm)
            .catch(this.handleRegistrationFailure)
            .then(this.reloadModelVersionsForCurrentRun)
            .catch(Utils.logErrorAndNotifyUser);
        } else {
          this.props
            .createModelVersionApi(
              selectedModelName,
              modelPath,
              runUuid,
              this.createModelVersionRequestId,
            )
            .then(this.resetAndClearModalForm)
            .catch(this.handleRegistrationFailure)
            .then(this.reloadModelVersionsForCurrentRun)
            .catch(Utils.logErrorAndNotifyUser);
        }
      }
    });
  };

  saveFormRef = (form) => {
    this.form = form;
  };

  saveFormComponentRef = (formComponent) => {
    this.formComponent = formComponent;
  };

  componentDidMount() {
    this.props.listRegisteredModelsApi();
  }

  componentDidUpdate(prevProps, prevState) {
    // Repopulate registered model list every time user launch the modal
    if (prevState.visible === false && this.state.visible === true) {
      this.props.listRegisteredModelsApi();
    }
  }

  render() {
    const { visible, confirmLoading } = this.state;
    const { disabled, modelByName } = this.props;
    return (
      <div className='register-model-btn-wrapper'>
        <Button
          className='register-model-btn'
          type='primary'
          onClick={this.showRegisterModal}
          disabled={disabled}
          htmlType='button'
        >
          Register Model
        </Button>
        <Modal
          title='Register Model'
          width={540}
          visible={visible}
          onOk={this.handleRegisterModel}
          okText='Register'
          confirmLoading={confirmLoading}
          onCancel={this.hideRegisterModal}
          centered
        >
          <RegisterModelForm
            modelByName={modelByName}
            ref={this.saveFormRef}
            wrappedComponentRef={this.saveFormComponentRef}
          />
        </Modal>
      </div>
    );
  }
}

const mapStateToProps = (state) => ({
  modelByName: state.entities.modelByName,
});

const mapDispatchToProps = {
  createRegisteredModelApi,
  createModelVersionApi,
  listRegisteredModelsApi,
  searchModelVersionsApi,
};

export default connect(mapStateToProps, mapDispatchToProps)(RegisterModelButton);
