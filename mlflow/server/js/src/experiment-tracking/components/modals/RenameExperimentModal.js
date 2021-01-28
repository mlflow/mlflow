import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import debounce from 'lodash/debounce';

import { GenericInputModal } from './GenericInputModal';
import { RenameForm, NEW_NAME_FIELD } from './RenameForm';
import { getExperimentNameValidator } from '../../../common/forms/validations';

import { updateExperimentApi, getExperimentApi } from '../../actions';
import { getExperiments } from '../../reducers/Reducers';

export class RenameExperimentModalImpl extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    experimentId: PropTypes.string,
    experimentName: PropTypes.string,
    experimentNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    onClose: PropTypes.func.isRequired,
    updateExperimentApi: PropTypes.func.isRequired,
    getExperimentApi: PropTypes.func.isRequired,
  };

  handleRenameExperiment = (values) => {
    // get value of input field
    const newExperimentName = values[NEW_NAME_FIELD];
    const updateExperimentPromise = this.props
      .updateExperimentApi(this.props.experimentId, newExperimentName)
      .then(() => this.props.getExperimentApi(this.props.experimentId));

    return updateExperimentPromise;
  };

  debouncedExperimentNameValidator = debounce(
    getExperimentNameValidator(() => this.props.experimentNames),
    400,
  );

  render() {
    const { isOpen, experimentName } = this.props;
    return (
      <GenericInputModal
        title='Rename Experiment'
        okText='Save'
        isOpen={isOpen}
        handleSubmit={this.handleRenameExperiment}
        onClose={this.props.onClose}
      >
        <RenameForm
          type='experiment'
          name={experimentName}
          visible={isOpen}
          validator={this.debouncedExperimentNameValidator}
        />
      </GenericInputModal>
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
  getExperimentApi,
};

export const RenameExperimentModal = connect(
  mapStateToProps,
  mapDispatchToProps,
)(RenameExperimentModalImpl);
