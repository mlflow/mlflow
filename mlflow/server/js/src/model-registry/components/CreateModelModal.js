import React from 'react';
import { GenericInputModal } from '../../experiment-tracking/components/modals/GenericInputModal';
import { CreateModelForm, MODEL_NAME_FIELD } from './CreateModelForm';
import { connect } from 'react-redux';
import { createRegisteredModelApi } from '../actions';
import { getUUID } from '../../common/utils/ActionUtils';
import { withRouter } from 'react-router-dom';
import { getModelPageRoute } from '../routes';
import { debounce } from 'lodash';
import PropTypes from 'prop-types';
import { modelNameValidator } from '../../common/forms/validations';
import { injectIntl } from 'react-intl';

export class CreateModelModalImpl extends React.Component {
  static propTypes = {
    createRegisteredModelApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
    modalVisible: PropTypes.bool.isRequired,
    hideModal: PropTypes.func.isRequired,
    // This allows for the modal to route back to previous page
    // on closing it when hitting cancel or exit button, not submit.
    navigateBackOnCancel: PropTypes.bool,
    intl: PropTypes.any,
  };

  createRegisteredModelRequestId = getUUID();

  handleCreateRegisteredModel = async (values) => {
    const result = await this.props.createRegisteredModelApi(
      values[MODEL_NAME_FIELD],
      this.createRegisteredModelRequestId,
    );
    const newModel = result.value && result.value.registered_model;
    if (newModel) {
      // Jump to the page of newly created model. Here we are yielding to next tick to allow modal
      // and form to finish closing and cleaning up.
      setTimeout(() => this.props.history.push(getModelPageRoute(newModel.name)));
    }
  };

  debouncedModelNameValidator = debounce(modelNameValidator, 400);

  handleOnCancel = () => {
    if (this.props.navigateBackOnCancel) {
      this.props.history.goBack();
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
        <CreateModelForm visible={modalVisible} validator={this.debouncedModelNameValidator} />
      </GenericInputModal>
    );
  }
}

const mapDispatchToProps = {
  createRegisteredModelApi,
};

export const CreateModelModal = withRouter(
  connect(undefined, mapDispatchToProps)(injectIntl(CreateModelModalImpl)),
);
