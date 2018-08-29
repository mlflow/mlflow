import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Immutable from 'immutable';
import {connect} from 'react-redux';

//import Dialog from '@material-ui/core/Dialog';
//import DialogContent from '@material-ui/core/DialogContent';
//import DialogTitle from '@material-ui/core/DialogTitle';

import RenameRunFormView from './RenameRunFormView';

import Utils from '../../utils/Utils';

import ReactModal from 'react-modal';
import { getRunTags, getApis } from '../../reducers/Reducers';

import { setTagApi, getUUID } from '../../Actions';

import RequestStateWrapper from '../RequestStateWrapper';



class RenameRunModal extends Component {
  constructor(props) {
    super(props);
    this.updateRunName = this.updateRunName.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  static propTypes = {
    open: PropTypes.bool,
    runUuid: PropTypes.string.isRequired,
    runTags: PropTypes.object.isRequired,
    dispatchSetTag: PropTypes.func.isRequired,
    onClose: PropTypes.func.isRequired,
  }

  updateRunName(obj) {
    const { newRunName } = obj;
    const tagKey = Utils.getRunTagName();
    const setTagRequestId = getUUID();
    const promise = this.props.dispatch(setTagApi(this.props.runUuid, tagKey, newRunName, setTagRequestId));
    return promise
  }

  handleSubmit = function(values) {
    const { runUuid } = this.props;
    const promise = this.updateRunName({...values, id: runUuid});
    return promise.then(function() { console.log("Sid: Promise resolved in RenameRunModal.js!")});
  }

  renderForm() {
    const { runUuid, runTags } = this.props;
    const runName = Utils.getRunDisplayName(runTags, runUuid);
    return <RenameRunFormView onSubmit={this.handleSubmit} initialValues={{ runName }}/>
  }

  render() {
    const { open } = this.props;
//    <RequestStateWrapper
//      requestIds={[this.state.getTagsRequestId]}
//    >
    return <ReactModal isOpen={open} role="dialog">{this.renderForm()}</ReactModal>
//    </RequestStateWrapper>
  }
}

function mapStateToProps(state, ownProps) {
  const { open, updateRunName, runUuid } = ownProps;
  const runTags = getRunTags(runUuid, state);
  return { open, updateRunName, runUuid, runTags };
}

// TODO: should this be higher up in the view hierarchy for reuse? Or maybe like a static thing
function onSetTag(tagKey, tagValue) {
  const setTagRequestId = this.props.dispatchSetTag(
    this.props.runUuid, tagKey, tagValue);
  this.setState({ setTagRequestId });
}

// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
    dispatchSetTag: (runUuid, tagKey, tagValue) => {
      const requestId = getUUID();
      dispatch(setTagApi(runUuid, tagKey, tagValue, requestId));
      return requestId;
    }
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(RenameRunModal)
