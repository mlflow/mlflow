import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';

import GenericInputModal from './GenericInputModal';

import { setTagApi, getUUID } from '../../Actions';
import Utils from '../../utils/Utils';

export class RenameRunModal extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    runUuid: PropTypes.string.isRequired,
    runName: PropTypes.string.isRequired,
    onClose: PropTypes.func.isRequired,
    setTagApi: PropTypes.func.isRequired,
  };

  handleRenameRun = (newRunName) => {
    const tagKey = Utils.runNameTag;
    const setTagRequestId = getUUID();

    return this.props.setTagApi(this.props.runUuid, tagKey, newRunName, setTagRequestId);
  }

  render() {
    const { isOpen, runName } = this.props;

    return (
      <GenericInputModal
        title='Rename Run'
        type='run'
        isOpen={isOpen}
        defaultValue={runName}
        handleSubmit={this.handleRenameRun}
        onClose={this.props.onClose}
        errorMessage='While renaming a run, an error occurred.'
      />
    );
  }
}

const mapDispatchToProps = {
  setTagApi,
};

export default connect(undefined, mapDispatchToProps)(RenameRunModal);

