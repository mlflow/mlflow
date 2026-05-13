/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { GenericInputModal } from '../../experiment-tracking/components/modals/GenericInputModal';
import { CreateModelForm, MODEL_NAME_FIELD } from './CreateModelForm';
import { connect } from 'react-redux';
import { createRegisteredModelApi } from '../actions';
import { getUUID } from '../../common/utils/ActionUtils';
import { ModelRegistryRoutes } from '../routes';
import { debounce } from 'lodash';
import { modelNameValidator } from '../../common/forms/validations';
import type { IntlShape } from 'react-intl';
import { injectIntl } from 'react-intl';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';

type Props = WithRouterNextProps & {
  createRegisteredModelApi: (...args: any[]) => any;
  modalVisible: boolean;
  hideModal: (...args: any[]) => any;
  navigateBackOnCancel?: boolean;
  intl: IntlShape;
};

class CreateModelModalImpl extends React.Component<Props> {
  createRegisteredModelRequestId = getUUID();

  handleCreateRegisteredModel = async (values: any) => {
    const result = await this.props.createRegisteredModelApi(
      values[MODEL_NAME_FIELD],
      this.createRegisteredModelRequestId,
    );
    const newModel = result.value && result.value.registered_model;
    if (newModel) {
      // Jump to the page of newly created model. Here we are yielding to next tick to allow modal
      // and form to finish closing and cleaning up.
      setTimeout(() => this.props.navigate(ModelRegistryRoutes.getModelPageRoute(newModel.name)));
    }
  };

  debouncedModelNameValidator = debounce(modelNameValidator, 400);

  handleOnCancel = () => {
    if (this.props.navigateBackOnCancel) {
      this.props.navigate(ModelRegistryRoutes.modelListPageRoute);
    }
  };

  render() {
    const { modalVisible, hideModal } = this.props;
    return (
      <GenericInputModal
        title={this.props.intl.formatMessage({
          defaultMessage: 'Create Model',
          description: 'Title text for creating model in the model registry',
        })}
        okText={this.props.intl.formatMessage({
          defaultMessage: 'Create',
          description: 'Create button text for creating model in the model registry',
        })}
        cancelText={this.props.intl.formatMessage({
          defaultMessage: 'Cancel',
          description: 'Cancel button text for creating model in the model registry',
        })}
        isOpen={modalVisible}
        handleSubmit={this.handleCreateRegisteredModel}
        onClose={hideModal}
        onCancel={this.handleOnCancel}
      >
        {/* @ts-expect-error TS(2322): Type '{ visible: boolean; validator: ((rule: any, ... Remove this comment to see the full error message */}
        <CreateModelForm visible={modalVisible} validator={this.debouncedModelNameValidator} />
      </GenericInputModal>
    );
  }
}

const mapDispatchToProps = {
  createRegisteredModelApi,
};

const CreateModelModalWithRouter = withRouterNext(
  connect(undefined, mapDispatchToProps)(injectIntl<'intl', Props>(CreateModelModalImpl)),
);

export const CreateModelModal = withErrorBoundary(ErrorUtils.mlflowServices.MODEL_REGISTRY, CreateModelModalWithRouter);
