/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import _ from 'lodash';
import { Modal, Button } from '@databricks/design-system';
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
  searchModelVersionsApi,
  searchRegisteredModelsApi,
} from '../actions';
import { connect } from 'react-redux';
import Utils from '../../common/utils/Utils';
import { getUUID } from '../../common/utils/ActionUtils';
import { getModelNameFilter } from '../../model-registry/utils/SearchUtils';

const MAX_SEARCH_REGISTERED_MODELS = 5;

type RegisterModelButtonImplProps = {
  disabled: boolean;
  runUuid: string;
  modelPath: string;
  modelRelativePath: string;
  modelByName: any;
  createRegisteredModelApi: (...args: any[]) => any;
  createModelVersionApi: (...args: any[]) => any;
  searchModelVersionsApi: (...args: any[]) => any;
  searchRegisteredModelsApi: (...args: any[]) => any;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
};

type RegisterModelButtonImplState = any; // used in drop-down list so not many are visible at once

export class RegisterModelButtonImpl extends React.Component<
  RegisterModelButtonImplProps,
  RegisterModelButtonImplState
> {
  form: any;

  state = {
    visible: false,
    confirmLoading: false,
    modelByName: {},
  };

  createRegisteredModelRequestId = getUUID();

  createModelVersionRequestId = getUUID();

  searchModelVersionRequestId = getUUID();

  constructor() {
    // @ts-expect-error TS(2554): Expected 1-2 arguments, but got 0.
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

  handleRegistrationFailure = (e: any) => {
    this.setState({ confirmLoading: false });
    Utils.logErrorAndNotifyUser(e);
  };

  handleSearchRegisteredModels = (input: any) => {
    this.props.searchRegisteredModelsApi(getModelNameFilter(input), MAX_SEARCH_REGISTERED_MODELS);
  };

  reloadModelVersionsForCurrentRun = () => {
    const { runUuid } = this.props;
    return this.props.searchModelVersionsApi({ run_id: runUuid }, this.searchModelVersionRequestId);
  };

  handleRegisterModel = () => {
    this.form.current.validateFields().then((values: any) => {
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
    this.props.searchRegisteredModelsApi();
  }

  componentDidUpdate(
    prevProps: RegisterModelButtonImplProps,
    prevState: RegisterModelButtonImplState,
  ) {
    // Repopulate registered model list every time user launch the modal
    if (prevState.visible === false && this.state.visible === true) {
      this.props.searchRegisteredModelsApi();
    }
  }

  renderRegisterModelForm() {
    const { modelByName } = this.props;
    return (
      <RegisterModelForm
        modelByName={modelByName}
        innerRef={this.form}
        onSearchRegisteredModels={_.debounce(this.handleSearchRegisteredModels, 300)}
      />
    );
  }

  renderFooter() {
    return [
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
    ];
  }

  render() {
    const { visible, confirmLoading } = this.state;
    const { disabled } = this.props;
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
          // @ts-expect-error TS(2322): Type '{ children: Element; title: any; width: numb... Remove this comment to see the full error message
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
          footer={this.renderFooter()}
        >
          {this.renderRegisterModelForm()}
        </Modal>
      </div>
    );
  }
}

const mapStateToProps = (state: any) => ({
  modelByName: state.entities.modelByName,
});

const mapDispatchToProps = {
  createRegisteredModelApi,
  createModelVersionApi,
  searchModelVersionsApi,
  searchRegisteredModelsApi,
};

// @ts-expect-error TS(2769): No overload matches this call.
export const RegisterModelButtonWithIntl = injectIntl(RegisterModelButtonImpl);
export const RegisterModelButton = connect(
  mapStateToProps,
  mapDispatchToProps,
)(RegisterModelButtonWithIntl);
