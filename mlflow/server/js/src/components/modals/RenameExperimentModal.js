import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import debounce from "lodash.debounce";

import GenericInputModal from './GenericInputModal';
import { RenameForm, NEW_NAME_FIELD } from './RenameForm';
import { getExperimentNameValidator } from './validation';

import { updateExperimentApi } from '../../Actions';
import { getExperiments } from '../../reducers/Reducers';


export class RenameExperimentModal extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    experimentId: PropTypes.number,
    experimentName: PropTypes.string,
    experimentNames: PropTypes.arrayOf(String).isRequired,
    onClose: PropTypes.func.isRequired,
    updateExperimentApi: PropTypes.func.isRequired,
  };

  handleRenameExperiment = (values) => {
    // get value of input field
    const newExperimentName = values[NEW_NAME_FIELD];
    return this.props.updateExperimentApi(this.props.experimentId, newExperimentName);
  };

  render() {
    const { isOpen, experimentName } = this.props;
    const experimentNameValidator = getExperimentNameValidator(this.props.experimentNames);

    const inputComponent = <RenameForm
      type='experiment'
      name={experimentName}
      visible={isOpen}
      validator={debounce(experimentNameValidator, 400)}
    />;

    return (
      <GenericInputModal
        title='Rename Experiment'
        childForm={inputComponent}
        isOpen={isOpen}
        handleSubmit={this.handleRenameExperiment}
        onClose={this.props.onClose}
      />
    );
  }
}

const mapStateToProps = (state) => {
  const experiments = getExperiments(state);
  const experimentNames = experiments.map((e) => e.getName());
  return { experimentNames };
};

const mapDispatchToProps = {
  updateExperimentApi,
};

export default connect(mapStateToProps, mapDispatchToProps)(RenameExperimentModal);

