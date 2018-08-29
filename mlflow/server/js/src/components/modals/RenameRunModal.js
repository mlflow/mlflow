import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Immutable from 'immutable';
import {connect} from 'react-redux';

//import Dialog from '@material-ui/core/Dialog';
//import DialogContent from '@material-ui/core/DialogContent';
//import DialogTitle from '@material-ui/core/DialogTitle';

import RenameRunFormView from './RenameRunFormView';

import Utils from '../../utils/Utils';

import { Button, Modal } from 'react-bootstrap';
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
    modalParams: PropTypes.instanceOf(Immutable.Map),
    open: PropTypes.bool,
    runUuid: PropTypes.string.isRequired,
    runTags: PropTypes.object.isRequired,
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
    const { runUuid, onClose } = this.props;
    const promise = this.updateRunName({...values, id: runUuid});
    return promise.then(onClose);
  }

  renderForm() {
    const { runUuid, runTags } = this.props;
    const runName = Utils.getRunDisplayName(runTags, runUuid);
    return <RenameRunFormView onSubmit={this.handleSubmit} initialValues={{ runName }}/>
  }

  render() {
    const { open } = this.props;
    return (<Modal show={open} onHide={this.props.onClose}>
      <Modal.Body>
      {this.renderForm()}
      <Button className="borderless-button" onClick={this.props.onClose} style={{position:"absolute", top: "8px", right: "8px"}}><i class="fas fa-times"/> </Button>
      </Modal.Body>
    </Modal>);
  }
}

function mapStateToProps(state, ownProps) {
  const { modalParams } = ownProps;
  const runUuid = modalParams.get('runUuid');
  const runTags = getRunTags(runUuid, state);
  return { runUuid, runTags };
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
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(RenameRunModal)
