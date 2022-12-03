import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Form, Select } from '@databricks/design-system';

const { Option } = Select;
export const EXPERIMENT_FIELD = 'experimentId';

/**
 * Component that renders a form for updating a run's or experiment's name.
 */
class MoveRunsFormComponent extends Component {
  static propTypes = {
    visible: PropTypes.bool.isRequired,
    experimentList: PropTypes.array.isRequired,
    innerRef: PropTypes.any.isRequired,
  };

  state = {
    selectedExperiment: null,
  };

  handleExperimentSelectChange = (selectedExperiment) => {
    this.setState({ selectedExperiment });
  };

  autoFocusInputRef = (inputToAutoFocus) => {
    this.inputToAutoFocus = inputToAutoFocus;
    inputToAutoFocus && inputToAutoFocus.focus();
    inputToAutoFocus && inputToAutoFocus.select();
  };

  autoFocus = (prevProps) => {
    if (prevProps.visible === false && this.props.visible === true) {
      // focus on input field
      this.inputToAutoFocus && this.inputToAutoFocus.focus();
      // select text
      this.inputToAutoFocus && this.inputToAutoFocus.select();
    }
  };

  render() {
    return (
      <Form ref={this.props.innerRef} layout='vertical'>
        <Form.Item
          name={EXPERIMENT_FIELD}
          rules={[{ required: true, message: `Please select an experiment.` }]}
          label={`Experiment name`}
        >
          <Select
            dropdownClassName='experiment-select-dropdown'
            onChange={this.handleExperimentSelectChange}
            placeholder='Move runs'
            filterOption={this.handleFilterOption}
            showSearch
          >
            {Object.values(this.props.experimentList).map((experiment) => (
              <Option value={experiment.experiment_id} key={experiment.experiment_id}>
                {experiment.name}
              </Option>
            ))}
          </Select>
        </Form.Item>
      </Form>
    );
  }
}

export const MoveRunsForm = MoveRunsFormComponent;
