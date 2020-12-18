import React from 'react';
import { Button } from 'antd';
import { GenericInputModal } from '../../experiment-tracking/components/modals/GenericInputModal';
import { CreateModelForm, MODEL_NAME_FIELD } from './CreateModelForm';
import { css } from 'emotion';
import { connect } from 'react-redux';
import { createRegisteredModelApi } from '../actions';
import { getUUID } from '../../common/utils/ActionUtils';
import { withRouter } from 'react-router-dom';
import { getModelPageRoute } from '../routes';
import { debounce } from 'lodash';
import PropTypes from 'prop-types';
import { modelNameValidator } from '../../common/forms/validations';

export class CreateModelButtonImpl extends React.Component {
  static propTypes = {
    createRegisteredModelApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
    buttonType: PropTypes.string,
    buttonText: PropTypes.string,
  };

  state = {
    modalVisible: false,
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

  hideModal = () => {
    this.setState({ modalVisible: false });
  };

  showModal = () => {
    this.setState({ modalVisible: true });
  };

  render() {
    const { modalVisible } = this.state;
    const buttonType = this.props.buttonType || 'primary';
    const buttonText = this.props.buttonText || 'Create Model';
    return (
      <div className={`create-model-btn-wrapper ${classNames.wrapper} ${modelClassNames}`}>
        <Button className={`create-model-btn`} type={buttonType} onClick={this.showModal}>
          {buttonText}
        </Button>
        <GenericInputModal
          title='Create Model'
          okText='Create'
          isOpen={modalVisible}
          handleSubmit={this.handleCreateRegisteredModel}
          onClose={this.hideModal}
        >
          <CreateModelForm visible={modalVisible} validator={this.debouncedModelNameValidator} />
        </GenericInputModal>
      </div>
    );
  }
}

const classNames = {
  wrapper: css({
    marginBottom: 24,
    display: 'inline',
  }),
};

const modelClassNames = css({
  '.ant-btn-link': {
    paddingLeft: 0,
    paddingRight: 0,
    color: '#2374BB',
  },
});

const mapDispatchToProps = {
  createRegisteredModelApi,
};

export const CreateModelButton = withRouter(
  connect(undefined, mapDispatchToProps)(CreateModelButtonImpl),
);
