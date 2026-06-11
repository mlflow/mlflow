/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { Alert, Modal, Spacer } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';

type Props = {
  okText?: string;
  cancelText?: string;
  isOpen?: boolean;
  onClose: (...args: any[]) => any;
  onCancel?: () => void;
  className?: string;
  footer?: React.ReactNode;
  handleSubmit: (...args: any[]) => any;
  title: React.ReactNode;
  okButtonProps?: React.ComponentProps<typeof Modal>['okButtonProps'];
};

type State = {
  isSubmitting: boolean;
  submissionError?: React.ReactNode;
};

/**
 * Generic modal that has a title and an input field with a save/submit button.
 * As of now, it is used to display the 'Rename Run' and 'Rename Experiment' modals.
 */
export class GenericInputModal extends Component<Props, State> {
  state: State = {
    isSubmitting: false,
    submissionError: undefined,
  };

  formRef = React.createRef();

  onSubmit = async () => {
    this.setState({ isSubmitting: true, submissionError: undefined });
    try {
      const values = await (this as any).formRef.current.validateFields();
      await this.props.handleSubmit(values);
      this.resetAndClearModalForm();
      this.onRequestCloseHandler();
    } catch (e) {
      this.handleSubmitFailure(e);
    }
  };

  resetAndClearModalForm = () => {
    this.setState({ isSubmitting: false, submissionError: undefined });
    (this as any).formRef.current.resetFields();
  };

  handleSubmitFailure = (e: any) => {
    // Form validation failures are already displayed inline on the fields
    // themselves, so no modal-level error is needed for them
    if (e?.errorFields) {
      this.setState({ isSubmitting: false });
      return;
    }
    // Keep submission failures in modal context (instead of a transient global
    // toast) so the user can read the error and correct their input
    const message = typeof e === 'string' ? e : e instanceof ErrorWrapper ? e.getMessageField() : e?.message;
    this.setState({
      isSubmitting: false,
      submissionError: message || (
        <FormattedMessage
          defaultMessage="The request failed. Please try again."
          description="Fallback error message shown inside an input modal when submission fails without details"
        />
      ),
    });
  };

  onRequestCloseHandler = () => {
    this.resetAndClearModalForm();
    this.props.onClose();
  };

  handleCancel = () => {
    this.onRequestCloseHandler();
    this.props.onCancel?.();
  };

  render() {
    const { isSubmitting, submissionError } = this.state;
    const { okText, cancelText, isOpen, footer, children, okButtonProps } = this.props;

    // add props (ref) to passed component
    const displayForm = React.Children.map(children, (child) => {
      // Checking isValidElement is the safe way and avoids a typescript
      // error too.
      if (React.isValidElement(child)) {
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
        okButtonProps={okButtonProps}
        onCancel={this.handleCancel}
        footer={footer}
        centered
      >
        {submissionError && (
          <>
            <Alert
              componentId="mlflow.generic_input_modal.submission_error"
              closable={false}
              message={submissionError}
              type="error"
            />
            <Spacer />
          </>
        )}
        {displayForm}
      </Modal>
    );
  }
}
