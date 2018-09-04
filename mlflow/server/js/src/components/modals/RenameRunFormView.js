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
    } = renderProps;
    const saveText = isSubmitting ? "Saving..." : "Save";
    return       <Modal.Dialog>
      <Modal.Title><h2 style={{"marginTop": 0}}>Rename Run</h2></Modal.Title>
      <Modal.Body style={{...styles.formField, "width": "100%"}}>
      <form onSubmit={handleSubmit} className="static-modal">
        <Field
            name="newRunName"
            label="New run name:"
            component={TextField}
            autoFocus
        />
      </form>;
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
      </Modal.Dialog>
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

const styles = {
  form: {
    width: 480,
  },
  buttonsDiv: {
    display: "flex",
    justifyContent: "flex-end",
  },
  formField: {
    marginBottom: 16,
  },
  label: {
    marginBottom: 4,
  },
};
