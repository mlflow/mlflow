/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import {
  LegacyForm,
  Input,
  PlusIcon,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxRoot,
  TypeaheadComboboxSectionHeader,
  TypeaheadComboboxStateChangeTypes,
  useComboboxState,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import './RegisterModelForm.css';

const CREATE_NEW_MODEL_LABEL = 'Create New Model';
// Keep the create action distinct from real model names in the form's submitted value.
export const CREATE_NEW_MODEL_OPTION_VALUE = `$$$__${CREATE_NEW_MODEL_LABEL}__$$$`;
export const SELECTED_MODEL_FIELD = 'selectedModel';
export const MODEL_NAME_FIELD = 'modelName';
const DESCRIPTION_FIELD = 'description';

interface RegisteredModelOption {
  name: string;
}

interface ModelComboboxProps {
  modelByName: Record<string, RegisteredModelOption>;
  onSearchRegisteredModels: (search: string) => void;
  onModelSelectChange: (selectedModel: string | null) => void;
  value?: string | null;
  onChange?: (value: string | null) => void;
  id?: string;
}

const modelOptionLabel = (modelName: string | null) =>
  modelName === CREATE_NEW_MODEL_OPTION_VALUE ? CREATE_NEW_MODEL_LABEL : (modelName ?? '');

function ModelCombobox({
  modelByName,
  onSearchRegisteredModels,
  onModelSelectChange,
  value,
  onChange,
  id,
}: ModelComboboxProps) {
  const allItems = React.useMemo<Array<string | null>>(
    () => [CREATE_NEW_MODEL_OPTION_VALUE, ...Object.values(modelByName ?? {}).map((model) => model.name)],
    [modelByName],
  );
  const [items, setItems] = React.useState(allItems);

  React.useEffect(() => setItems(allItems), [allItems]);

  const comboboxState = useComboboxState<string | null>({
    componentId: 'mlflow.register_model_form.model',
    allItems,
    items,
    setItems,
    itemToString: modelOptionLabel,
    matcher: (modelName, query) => modelOptionLabel(modelName).toLowerCase().includes(query),
    formValue: value ?? null,
    formOnChange: (selectedModel) => {
      // The older combobox invokes this callback from its reducer, so update the form after render.
      Promise.resolve().then(() => {
        onChange?.(selectedModel);
        onModelSelectChange(selectedModel);
      });
    },
    onIsOpenChange: ({ isOpen }) => {
      if (isOpen) setItems(allItems);
    },
    onStateChange: (change) => {
      if (change.type === TypeaheadComboboxStateChangeTypes.InputChange && change.inputValue !== undefined) {
        onSearchRegisteredModels(change.inputValue);
      }
    },
    preventUnsetOnBlur: true,
  });
  const filteredModels = items.filter(
    (modelName): modelName is string => modelName !== null && modelName !== CREATE_NEW_MODEL_OPTION_VALUE,
  );
  const showCreateOption = items.includes(CREATE_NEW_MODEL_OPTION_VALUE);

  return (
    <TypeaheadComboboxRoot
      aria-label="Model"
      comboboxState={comboboxState}
      onClick={(event) => {
        if (event.target === event.currentTarget) comboboxState.toggleMenu();
      }}
    >
      <TypeaheadComboboxInput
        id={id}
        aria-label="Model"
        placeholder="Select a model"
        comboboxState={comboboxState}
        showComboboxToggleButton
      />
      <TypeaheadComboboxMenu comboboxState={comboboxState} matchTriggerWidth maxHeight={280}>
        {showCreateOption && (
          <TypeaheadComboboxMenuItem item={CREATE_NEW_MODEL_OPTION_VALUE} index={0} comboboxState={comboboxState}>
            <PlusIcon /> {CREATE_NEW_MODEL_LABEL}
          </TypeaheadComboboxMenuItem>
        )}
        {filteredModels.length > 0 && <TypeaheadComboboxSectionHeader>Models</TypeaheadComboboxSectionHeader>}
        {filteredModels.map((modelName, index) => (
          <TypeaheadComboboxMenuItem
            item={modelName}
            key={modelName}
            index={index + (showCreateOption ? 1 : 0)}
            comboboxState={comboboxState}
          >
            {modelName}
          </TypeaheadComboboxMenuItem>
        ))}
      </TypeaheadComboboxMenu>
    </TypeaheadComboboxRoot>
  );
}

type Props = {
  modelByName?: any;
  isCopy?: boolean;
  onSearchRegisteredModels: (...args: any[]) => any;
  innerRef: any;
};

type State = any;

export class RegisterModelForm extends React.Component<Props, State> {
  state = {
    selectedModel: null,
  };

  handleModelSelectChange = (selectedModel: any) => {
    this.setState({ selectedModel });
  };

  modelNameValidator = (rule: any, value: any, callback: any) => {
    const { modelByName } = this.props;
    callback(modelByName[value] ? `Model "${value}" already exists.` : undefined);
  };

  renderExplanatoryText() {
    const { isCopy } = this.props;
    const { selectedModel } = this.state;
    const creatingNewModel = selectedModel === CREATE_NEW_MODEL_OPTION_VALUE;

    if (!selectedModel || creatingNewModel) {
      return null;
    }

    const explanation = isCopy ? (
      <FormattedMessage
        defaultMessage="The model version will be copied to {selectedModel} as a new version."
        description="Model registry > OSS Promote model modal > copy explanatory text"
        values={{ selectedModel: selectedModel }}
      />
    ) : (
      <FormattedMessage
        defaultMessage="The model will be registered as a new version of {selectedModel}."
        description="Explantory text for registering a model"
        values={{ selectedModel: selectedModel }}
      />
    );

    return (
      <p className="modal-explanatory-text" css={(theme) => ({ color: theme.colors.textSecondary })}>
        {explanation}
      </p>
    );
  }

  render() {
    const { modelByName, innerRef, isCopy } = this.props;
    const { selectedModel } = this.state;
    const creatingNewModel = selectedModel === CREATE_NEW_MODEL_OPTION_VALUE;
    return (
      // @ts-expect-error TS(2322): Type '{ children: (Element | null)[]; ref: any; la... Remove this comment to see the full error message
      <LegacyForm ref={innerRef} layout="vertical" className="mlflow-register-model-form">
        {/* "+ Create new model" OR "Select existing model" */}
        <LegacyForm.Item
          label={isCopy ? <b>Copy to model</b> : 'Model'}
          name={SELECTED_MODEL_FIELD}
          rules={[{ required: true, message: 'Please select a model or create a new one.' }]}
        >
          <ModelCombobox
            modelByName={modelByName}
            onSearchRegisteredModels={this.props.onSearchRegisteredModels}
            onModelSelectChange={this.handleModelSelectChange}
          />
        </LegacyForm.Item>

        {/* Name the new model when "+ Create new model" is selected */}
        {creatingNewModel ? (
          <LegacyForm.Item
            label="Model Name"
            name={MODEL_NAME_FIELD}
            rules={[
              { required: true, message: 'Please input a name for the new model.' },
              { validator: this.modelNameValidator },
            ]}
          >
            <Input
              componentId="codegen_mlflow_app_src_model-registry_components_registermodelform.tsx_132"
              placeholder="Input a model name"
            />
          </LegacyForm.Item>
        ) : null}

        {/* Explanatory text shown when existing model is selected */}
        {this.renderExplanatoryText()}
      </LegacyForm>
    );
  }
}
