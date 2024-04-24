/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { Modal } from '@databricks/design-system';

import Utils from '../../../common/utils/Utils';

type Props = {
  okText?: string;
  cancelText?: string;
  isOpen?: boolean;
  onClose: (...args: any[]) => any;
  onCancel?: (...args: any[]) => any;
  className?: string;
  footer?: React.ReactNode;
  handleSubmit: (...args: any[]) => any;
  title: React.ReactNode;
};

type State = any;

/**
 * Generic modal that has a title and an input field with a save/submit button.
 * As of now, it is used to display the 'Rename Run' and 'Rename Experiment' modals.
 */
export class GenericInputModal extends Component<Props, State> {
  state = {
    isSubmitting: false,
  };

  formRef = React.createRef();

  onSubmit = async () => {
    this.setState({ isSubmitting: true });
    try {
      const values = await (this as any).formRef.current.validateFields();

      // call handleSubmit from parent component, pass form values
      // handleSubmit is expected to return a promise
      return await this.props
        .handleSubmit(values)
        .then(this.resetAndClearModalForm)
        .catch(this.handleSubmitFailure)
        .finally(this.onRequestCloseHandler);
    } catch (e) {
      this.setState({ isSubmitting: false });
      return Promise.reject(e);
    }
  };

  resetAndClearModalForm = () => {
    this.setState({ isSubmitting: false });
    (this as any).formRef.current.resetFields();
  };

  handleSubmitFailure = (e: any) => {
    this.setState({ isSubmitting: false });
    Utils.logErrorAndNotifyUser(e);
  };

  onRequestCloseHandler = () => {
    if (!this.state.isSubmitting) {
      this.resetAndClearModalForm();
      this.props.onClose();
    }
  };

  handleCancel = () => {
    this.onRequestCloseHandler();
    // Check for optional `onCancel` method prop.
    if (this.props.onCancel !== undefined) {
      this.props.onCancel();
    }
  };

  render() {
    const { isSubmitting } = this.state;
    const { okText, cancelText, isOpen, footer, children } = this.props;

    // add props (ref) to passed component
    const displayForm = React.Children.map(children, (child) => {
      // Checking isValidElement is the safe way and avoids a typescript
      // error too.
      if (React.isValidElement(child)) {
        // @ts-expect-error TODO: fix this
        return React.cloneElement(child, { innerRef: this.formRef });
      }
      return child;
    });

    return (
      <Modal
        data-testid="mlflow-input-modal"
        className={this.props.className}
        title={this.props.title}
        // @ts-expect-error TS(2322): Type '{ children: {}[] | null | undefined; "data-t... Remove this comment to see the full error message
        width={540}
        visible={isOpen}
        onOk={this.onSubmit}
        okText={okText}
        cancelText={cancelText}
        confirmLoading={isSubmitting}
        onCancel={this.handleCancel}
        footer={footer}
        centered
      >
        {displayForm}
      </Modal>
    );
  }
}
