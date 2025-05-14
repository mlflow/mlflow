/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';

import { LegacyForm, Input } from '@databricks/design-system';

export const NEW_NAME_FIELD = 'newName';

type Props = {
  type: string;
  name: string;
  visible: boolean;
  validator?: (...args: any[]) => any;
  innerRef: any;
};

/**
 * Component that renders a form for updating a run's or experiment's name.
 */
class RenameFormComponent extends Component<Props> {
  inputToAutoFocus: any;

  componentDidUpdate(prevProps: Props) {
    this.autoFocus(prevProps);
    this.resetFields(prevProps);
  }

  autoFocusInputRef = (inputToAutoFocus: any) => {
    this.inputToAutoFocus = inputToAutoFocus;
    inputToAutoFocus && inputToAutoFocus.focus();
    inputToAutoFocus && inputToAutoFocus.select();
  };

  autoFocus = (prevProps: any) => {
    if (prevProps.visible === false && this.props.visible === true) {
      // focus on input field
      this.inputToAutoFocus && this.inputToAutoFocus.focus();
      // select text
      this.inputToAutoFocus && this.inputToAutoFocus.select();
    }
  };

  resetFields = (prevProps: any) => {
    const formRef = this.props.innerRef;
    if (prevProps.name !== this.props.name) {
      // reset input field to reset displayed initialValue
      formRef.current.resetFields([NEW_NAME_FIELD]);
    }
  };

  render() {
    return (
      // @ts-expect-error TS(2322): Type '{ children: Element; ref: any; layout: "vert... Remove this comment to see the full error message
      <LegacyForm ref={this.props.innerRef} layout="vertical">
        <LegacyForm.Item
          name={NEW_NAME_FIELD}
          initialValue={this.props.name}
          rules={[
            { required: true, message: `Please input a new name for the ${this.props.type}.` },
            { validator: this.props.validator },
          ]}
          label={`New ${this.props.type} name`}
        >
          <Input
            componentId="codegen_mlflow_app_src_experiment-tracking_components_modals_renameform.tsx_69"
            placeholder={`Input a ${this.props.type} name`}
            ref={this.autoFocusInputRef}
            data-testid="rename-modal-input"
          />
        </LegacyForm.Item>
      </LegacyForm>
    );
  }
}

export const RenameForm = RenameFormComponent;
