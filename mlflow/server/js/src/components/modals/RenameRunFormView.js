import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Button, Modal } from 'react-bootstrap';
import { withRouter } from 'react-router-dom';
import { Formik, Field } from 'formik';

import { validationSchema } from './validation';
import TextField from './TextField';

/**
 * Component that renders a form for updating a run's name. Expects to be 'closeable'
 * (i.e. rendered within a closeable dialog) and so accepts an `onClose` callback.
 */
class RenameRunFormView extends Component {
  static propTypes = {
    onSubmit: PropTypes.func.isRequired,
    onClose: PropTypes.func.isRequired,
    runName: PropTypes.string.isRequired,
    experimentId: PropTypes.number.isRequired
  }

  renderForm = (renderProps) => {
    const {
      handleSubmit,
      isSubmitting,
      status,
    } = renderProps;
    const saveText = isSubmitting ? "Saving..." : "Save";

    return <form onSubmit={handleSubmit} className="rename-run-form">
      <Modal.Header><Modal.Title>Rename Run</Modal.Title></Modal.Header>
      <Modal.Body>
        { status && status.errorMsg &&
          <div className="text-danger">
            <i className="fas fa-exclamation-triangle"></i>
            {status.errorMsg}
          </div>
        }
        <Field
            name="newRunName"
            label="New run name:"
            autoFocus
            component={TextField}
        />
      </Modal.Body>
      <Modal.Footer>
        <Button
          bsStyle="default"
          className="cancel-button"
          disabled={isSubmitting}
          onClick={this.props.onClose}
        >
          Cancel
        </Button>
        <Button bsStyle="primary" type="submit" className="save-button" disabled={isSubmitting}>
          {saveText}
        </Button>
      </Modal.Footer>
    </form>;
  }

  render() {
    return (<div>
      <Formik
        initialValues={{newRunName: this.props.runName}}
        validationSchema={validationSchema}
        onSubmit={this.props.onSubmit}
        render={this.renderForm}/>
      </div>);
  }
}

export default withRouter(RenameRunFormView);
