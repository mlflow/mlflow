import React from 'react';
import _ from 'lodash';
import { Modal, Button } from 'antd';
import { FormattedMessage, injectIntl } from 'react-intl';

import {
  RegisterModelForm,
  CREATE_NEW_MODEL_OPTION_VALUE,
  SELECTED_MODEL_FIELD,
  MODEL_NAME_FIELD,
} from './RegisterModelForm';
import {
  createRegisteredModelApi,
  createModelVersionApi,
  listModelStagesApi,
  listRegisteredModelsApi,
  searchModelVersionsApi,
  searchRegisteredModelsApi,
} from '../actions';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import Utils from '../../common/utils/Utils';
import { getUUID } from '../../common/utils/ActionUtils';
import { getModelNameFilter } from '../../model-registry/utils/SearchUtils';

const MAX_SEARCH_REGISTERED_MODELS = 5; // used in drop-down list so not many are visible at once

export class RegisterModelButtonImpl extends React.Component {
  static propTypes = {
    // own props
    disabled: PropTypes.bool.isRequired,
    runUuid: PropTypes.string.isRequired,
    modelPath: PropTypes.string,
    // connected props
    modelByName: PropTypes.object.isRequired,
    createRegisteredModelApi: PropTypes.func.isRequired,
    createModelVersionApi: PropTypes.func.isRequired,
    listModelStagesApi: PropTypes.func.isRequired,
    listRegisteredModelsApi: PropTypes.func.isRequired,
    searchModelVersionsApi: PropTypes.func.isRequired,
    searchRegisteredModelsApi: PropTypes.func.isRequired,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  state = {
    visible: false,
    confirmLoading: false,
    modelByName: {},
  };

  createRegisteredModelRequestId = getUUID();

  createModelVersionRequestId = getUUID();

  searchModelVersionRequestId = getUUID();

  constructor() {
    super();
    this.form = React.createRef();
  }

  showRegisterModal = () => {
    this.setState({ visible: true });
  };

  hideRegisterModal = () => {
    this.setState({ visible: false });
  };

  resetAndClearModalForm = () => {
    this.setState({ visible: false, confirmLoading: false });
    this.form.current.resetFields();
  };

  handleRegistrationFailure = (e) => {
    this.setState({ confirmLoading: false });
    Utils.logErrorAndNotifyUser(e);
  };

  handleSearchRegisteredModels = (input) => {
    this.props.searchRegisteredModelsApi(getModelNameFilter(input), MAX_SEARCH_REGISTERED_MODELS);
  };

  reloadModelVersionsForCurrentRun = () => {
    const { runUuid } = this.props;
    return this.props.searchModelVersionsApi({ run_id: runUuid }, this.searchModelVersionRequestId);
  };

  handleRegisterModel = () => {
    this.form.current.validateFields().then((values) => {
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
    });
  };

  componentDidMount() {
    this.props.listModelStagesApi();
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
          <FormattedMessage
            defaultMessage='Register Model'
            description='Button text to register the model for deployment'
          />
        </Button>
        <Modal
          title={this.props.intl.formatMessage({
            defaultMessage: 'Register Model',
            description: 'Register model modal title to register the model for deployment',
          })}
          width={540}
          visible={visible}
          onOk={this.handleRegisterModel}
          okText={this.props.intl.formatMessage({
            defaultMessage: 'Register',
            description: 'Confirmation text to register the model',
          })}
          confirmLoading={confirmLoading}
          onCancel={this.hideRegisterModal}
          centered
          footer={[
            <Button key='back' onClick={this.hideRegisterModal}>
              <FormattedMessage
                defaultMessage='Cancel'
                description='Cancel button text to cancel the flow to register the model'
              />
            </Button>,
            <Button
              key='submit'
              type='primary'
              onClick={this.handleRegisterModel}
              data-test-id='confirm-register-model'
            >
              <FormattedMessage
                defaultMessage='Register'
                description='Register button text to register the model'
              />
            </Button>,
          ]}
        >
          <RegisterModelForm
            modelByName={modelByName}
            innerRef={this.form}
            onSearchRegisteredModels={_.debounce(this.handleSearchRegisteredModels, 300)}
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
  listModelStagesApi,
  listRegisteredModelsApi,
  searchModelVersionsApi,
  searchRegisteredModelsApi,
};

export const RegisterModelButtonWithIntl = injectIntl(RegisterModelButtonImpl);
export const RegisterModelButton = connect(
  mapStateToProps,
  mapDispatchToProps,
)(RegisterModelButtonWithIntl);
