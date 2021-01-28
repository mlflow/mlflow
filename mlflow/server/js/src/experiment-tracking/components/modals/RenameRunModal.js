import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';

import { GenericInputModal } from './GenericInputModal';
import { RenameForm, NEW_NAME_FIELD } from './RenameForm';

import { setTagApi } from '../../actions';
import Utils from '../../../common/utils/Utils';
import { getUUID } from '../../../common/utils/ActionUtils';

export class RenameRunModalImpl extends Component {
  static propTypes = {
    isOpen: PropTypes.bool,
    runUuid: PropTypes.string.isRequired,
    runName: PropTypes.string.isRequired,
    onClose: PropTypes.func.isRequired,
    setTagApi: PropTypes.func.isRequired,
  };

  handleRenameRun = (values) => {
    // get value of input field
    const newRunName = values[NEW_NAME_FIELD];

    const tagKey = Utils.runNameTag;
    const setTagRequestId = getUUID();

    return this.props.setTagApi(this.props.runUuid, tagKey, newRunName, setTagRequestId);
  };

  render() {
    const { isOpen, runName } = this.props;
    return (
      <GenericInputModal
        title='Rename Run'
        okText='Save'
        isOpen={isOpen}
        handleSubmit={this.handleRenameRun}
        onClose={this.props.onClose}
      >
        <RenameForm type='run' name={runName} visible={isOpen} />
      </GenericInputModal>
    );
  }
}

const mapDispatchToProps = {
  setTagApi,
};

export const RenameRunModal = connect(undefined, mapDispatchToProps)(RenameRunModalImpl);
