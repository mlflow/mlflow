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

const modalStyles = {
  content : {
    top: '50%',
    left: '50%',
    right: 'auto',
    bottom: 'auto',
    marginRight: '-50%',
    transform: 'translate(-50%, -50%)'
  },
  overlay: {
    backgroundColor: 'rgba(33, 37, 41, .75)'
  }
};

class RenameRunModal extends Component {
  constructor(props) {
    super(props);
    this.updateRunName = this.updateRunName.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.onRequestCloseHandler = this.onRequestCloseHandler.bind(this);
  }

  state = {
    isSubmittingState: false,
  }

  static propTypes = {
    open: PropTypes.bool,
    experimentId: PropTypes.number.isRequired,
    runUuid: PropTypes.string.isRequired,
    runTags: PropTypes.object.isRequired,
    onClose: PropTypes.func.isRequired,
  }

  updateRunName(newRunName) {
    const tagKey = Utils.runNameTag;
    const setTagRequestId = getUUID();
    return this.props.dispatch(
      setTagApi(this.props.runUuid, tagKey, newRunName, setTagRequestId));
  }

  handleSubmit = function(newRunName) {
    const { runUuid, onClose } = this.props;
    this.setState({isSubmittingState: true});
    // We don't close the modal here, instead delegating that logic to the the form view component,
    // which is responsible for waiting on the promise & calling a callback to close the
    // modal once submission completes. This allows us to avoid removing the form component from
    // underneath itself
    return this.updateRunName(newRunName).finally(function() {
      this.setState({isSubmittingState: false});
    }.bind(this));
  }

  renderForm() {
    const { runUuid, runTags, onClose, experimentId } = this.props;
    const runName = Utils.getRunName(runTags, runUuid);
    return <RenameRunFormView onSubmit={this.handleSubmit}
      onClose={this.onRequestCloseHandler} runName={runName} experimentId={experimentId}/>
  }

  onRequestCloseHandler(event) {
    if (!this.state.isSubmittingState) {
      this.props.onClose();
    }
  }

  render() {
    const { open } = this.props;
    return (
    <ReactModal isOpen={open} onRequestClose={this.onRequestCloseHandler} style={modalStyles}
     appElement={document.body}>
      {this.renderForm()}
      <a className="exit-link"><i onClick={this.onRequestCloseHandler} className="fas fa-times"/></a>
    </ReactModal>);
  }
}

function mapStateToProps(state, ownProps) {
  const { runUuid } = ownProps;
  const runTags = getRunTags(runUuid, state);
  return { runTags };
}


// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(RenameRunModal)
