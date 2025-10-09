/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { identity, isUndefined, debounce } from 'lodash';
import type { ButtonProps } from '@databricks/design-system';
import { Button, Modal, Spacer, LegacyTooltip, Typography, ModalProps } from '@databricks/design-system';
import { FormattedMessage, injectIntl, type IntlShape } from 'react-intl';
import {
  CREATE_NEW_MODEL_OPTION_VALUE,
  MODEL_NAME_FIELD,
  RegisterModelForm,
  SELECTED_MODEL_FIELD,
} from './RegisterModelForm';
import {
  createModelVersionApi,
  createRegisteredModelApi,
  searchModelVersionsApi,
  searchRegisteredModelsApi,
} from '../actions';
import { connect } from 'react-redux';
import Utils from '../../common/utils/Utils';
import { getUUID } from '../../common/utils/ActionUtils';
import { getModelNameFilter } from '../utils/SearchUtils';

const MAX_SEARCH_REGISTERED_MODELS = 5;

type RegisterModelImplProps = {
  disabled: boolean;
  runUuid?: string;
  loggedModelId?: string;
  modelPath: string;
  modelRelativePath?: string;
  modelByName: any;
  createRegisteredModelApi: (...args: any[]) => any;
  createModelVersionApi: (...args: any[]) => any;
  searchModelVersionsApi: (...args: any[]) => any;
  searchRegisteredModelsApi: (...args: any[]) => any;
  intl: IntlShape;
  /**
   * Type of button to display ("primary", "link", etc.)
   */
  buttonType?: ButtonProps['type'];
  /**
   * Tooltip to display on hover
   */
  tooltip?: React.ReactNode;
  /**
   * Whether to show the button. If set to true, only modal will be used and button will not be shown.
   */
  showButton?: boolean;
  /**
   * Whether the modal is visible. If set, modal visibility will be controlled by the props.
   */
  modalVisible?: boolean;
  /**
   * Callback to close the modal. If set, modal visibility will be controlled by the parent component.
   */
  onCloseModal?: () => void;
  /**
   * Callback to run after the model is registered.
   */
  onRegisterSuccess?: (data?: { value: { status?: string } }) => void;
  /**
   * Callback to run after the model is registered.
   */
  onRegisterFailure?: (reason?: any) => void;
};

type RegisterModelImplState = any; // used in drop-down list so not many are visible at once

/**
 * Component with a set of controls used to register a logged model.
 * Includes register modal and optional "Register" button.
 */
export class RegisterModelImpl extends React.Component<RegisterModelImplProps, RegisterModelImplState> {
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
    this.props.onCloseModal?.();
  };

  resetAndClearModalForm = () => {
    this.setState({ visible: false, confirmLoading: false });
    this.form.current?.resetFields();
    this.props.onCloseModal?.();
  };

  handleRegistrationFailure = (e: any) => {
    this.setState({ confirmLoading: false });
    Utils.logErrorAndNotifyUser(e);
  };

  handleSearchRegisteredModels = (input: any) => {
    if (this.isWorkspaceModelRegistryEnabled) {
      this.props.searchRegisteredModelsApi(getModelNameFilter(input), MAX_SEARCH_REGISTERED_MODELS);
    }
  };

  reloadModelVersionsForCurrentRun = () => {
    const { runUuid } = this.props;
    return this.props.searchModelVersionsApi({ run_id: runUuid }, this.searchModelVersionRequestId);
  };

  handleRegisterModel = () => {
    return this.form.current.validateFields().then((values: any) => {
      this.setState({ confirmLoading: true });
      const { runUuid, modelPath } = this.props;
      const selectedModelName = values[SELECTED_MODEL_FIELD];
      if (selectedModelName === CREATE_NEW_MODEL_OPTION_VALUE) {
        // When user choose to create a new registered model during the registration, we need to
        // 1. Create a new registered model
        // 2. Create model version #1 in the new registered model
        return this.props
          .createRegisteredModelApi(values[MODEL_NAME_FIELD], this.createRegisteredModelRequestId)
          .then(() =>
            this.props.createModelVersionApi(
              values[MODEL_NAME_FIELD],
              modelPath,
              runUuid,
              [],
              this.createModelVersionRequestId,
              this.props.loggedModelId,
            ),
          )
          .then(this.props.onRegisterSuccess ?? identity)
          .then(this.resetAndClearModalForm)
          .catch(this.props.onRegisterFailure ?? this.handleRegistrationFailure)
          .then(this.reloadModelVersionsForCurrentRun)
          .catch(Utils.logErrorAndNotifyUser);
      } else {
        return this.props
          .createModelVersionApi(
            selectedModelName,
            modelPath,
            runUuid,
            [],
            this.createModelVersionRequestId,
            this.props.loggedModelId,
          )
          .then(this.props.onRegisterSuccess ?? identity)
          .then(this.resetAndClearModalForm)
          .catch(this.props.onRegisterFailure ?? this.handleRegistrationFailure)
          .then(this.reloadModelVersionsForCurrentRun)
          .catch(Utils.logErrorAndNotifyUser);
      }
    });
  };

  get isWorkspaceModelRegistryEnabled() {
    return true;
  }

  componentDidMount() {
    if (this.isWorkspaceModelRegistryEnabled) {
      this.props.searchRegisteredModelsApi();
    }
  }

  componentDidUpdate(prevProps: RegisterModelImplProps, prevState: RegisterModelImplState) {
    // Repopulate registered model list every time user launch the modal
    if (prevState.visible === false && this.state.visible === true && this.isWorkspaceModelRegistryEnabled) {
      this.props.searchRegisteredModelsApi();
    }
  }
  renderRegisterModelForm() {
    const { modelByName } = this.props;
    return (
      <RegisterModelForm
        modelByName={modelByName}
        innerRef={this.form}
        onSearchRegisteredModels={debounce(this.handleSearchRegisteredModels, 300)}
      />
    );
  }

  renderFooter() {
    return [
      <Button
        componentId="codegen_mlflow_app_src_model-registry_components_registermodel.tsx_242"
        key="back"
        onClick={this.hideRegisterModal}
      >
        <FormattedMessage
          defaultMessage="Cancel"
          description="Cancel button text to cancel the flow to register the model"
        />
      </Button>,
      <Button
        componentId="codegen_mlflow_app_src_model-registry_components_registermodel.tsx_248"
        key="submit"
        type="primary"
        onClick={() => this.handleRegisterModel()}
        data-testid="confirm-register-model"
      >
        <FormattedMessage defaultMessage="Register" description="Register button text to register the model" />
      </Button>,
    ];
  }

  renderHelper(disableButton: boolean, form: React.ReactNode, footer: React.ReactNode) {
    const { visible, confirmLoading } = this.state;
    const { showButton = true, buttonType } = this.props;
    return (
      <div className="register-model-btn-wrapper">
        {showButton && (
          <LegacyTooltip title={this.props.tooltip || null} placement="left">
            <Button
              componentId="codegen_mlflow_app_src_model-registry_components_registermodel.tsx_261"
              className="register-model-btn"
              type={buttonType}
              onClick={this.showRegisterModal}
              disabled={disableButton}
              htmlType="button"
            >
              <FormattedMessage
                defaultMessage="Register model"
                description="Button text to register the model for deployment"
              />
            </Button>
          </LegacyTooltip>
        )}
        <Modal
          title={this.props.intl.formatMessage({
            defaultMessage: 'Register model',
            description: 'Register model modal title to register the model for deployment',
          })}
          // @ts-expect-error TS(2322): Type '{ children: Element; title: any; width: numb... Remove this comment to see the full error message
          width={540}
          visible={this.props.modalVisible || visible}
          onOk={() => this.handleRegisterModel()}
          okText={this.props.intl.formatMessage({
            defaultMessage: 'Register',
            description: 'Confirmation text to register the model',
          })}
          confirmLoading={confirmLoading}
          onCancel={this.hideRegisterModal}
          centered
          footer={footer}
        >
          {form}
        </Modal>
      </div>
    );
  }

  render() {
    const { disabled } = this.props;
    return this.renderHelper(disabled, this.renderRegisterModelForm(), this.renderFooter());
  }
}

const mapStateToProps = (state: any) => {
  return {
    modelByName: state.entities.modelByName,
  };
};

const mapDispatchToProps = {
  createRegisteredModelApi,
  createModelVersionApi,
  searchModelVersionsApi,
  searchRegisteredModelsApi,
};

export const RegisterModelWithIntl = injectIntl(RegisterModelImpl);
export const RegisterModel = connect(mapStateToProps, mapDispatchToProps)(RegisterModelWithIntl);

// ..
