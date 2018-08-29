import React, { Component } from 'react'
import PropTypes from 'prop-types';
import { connect } from 'react-redux';

import { Formik, Field } from 'formik';

import { withStyles } from '@material-ui/core/styles';
import * as AntdFields from './fields/antd';
import * as BootstrapFields from './fields/bootstrap';
import * as MaterialFields from './fields/material';
import { ANTD, BOOTSTRAP, MATERIAL_UI } from '../../uiComponentLibrary/actions';
import {getComponentLibrary} from '../../uiComponentLibrary/reducer';


import { validationSchema } from '../validation';

const styles = theme => ({
  root: {
    margin: 'auto',
    width: 600,
  },
  field: {
    width: 250,
    marginTop: 5
  }
});

class FormikContactForm extends Component {

  static propTypes = {
    componentLibrary: PropTypes.string,
    onSubmit: PropTypes.func.isRequired
  }

  getUiComponents() {
    const { componentLibrary } = this.props;
    switch(componentLibrary) {
      case ANTD:
        return AntdFields;
      case BOOTSTRAP:
        return BootstrapFields;
      case MATERIAL_UI:
        return MaterialFields;
      default:
        throw new Error('unknown component library')
    }
  }

  getInputProps(renderProps, name, label) {
    return {
      name,
      label,
      onChange: renderProps.handleChange,
      onBlur: renderProps.handleBlur,
      value: renderProps.values[name],
      error: renderProps.errors[name],
      touched: renderProps.touched[name],
    }
  }

  handleSubmit = (
    values,
    {
      props,
      setSubmitting,
      setErrors /* setValues, setStatus, and other goodies */,
    }) => {
      return this.props.onSubmit(values).catch((err) => {
        setErrors(err.errors)
      }).finally(() => {
        setSubmitting(false)
      })
    }

  renderForm = (renderProps) => {
    const {
      handleSubmit,
      isSubmitting,
    } = renderProps;
    const { classes } = this.props;
    const { TextField, Select, Checkbox } = this.getUiComponents();
    return <form onSubmit={handleSubmit}>
      <div>
        <Field
          component={TextField}
          name="firstName"
          label="First Name"
          autoFocus
          className={classes.field}
        />
      </div>
      <div>
        <Field
          component={TextField}
          name="lastName"
          label="Last Name"
          className={classes.field}
        />
      </div>
      <div>
        <Field
          component={TextField}
          name="age"
          label="Age"
          className={classes.field}
        />
      </div>
      <div>
        <Field
          component={TextField}
          name="email"
          label="Email"
          className={classes.field}
        />
      </div>
      <div>
        <Field
          component={Select}
          name="favoriteColor"
          label="Favorite Color"
          options={[
            {label: 'Red', value: 'ff0000'},
            {label: 'Green', value: '00ff00'},
            {label: 'Blue', value: '0000ff'},
          ]}
          className={classes.field}
        />
      </div>
      <div>
        <Field
          component={Checkbox}
          name="employed"
          label="Employed"
          className={classes.field}
        />
      </div>
      <button type="submit" disabled={isSubmitting}>
        Submit
      </button>
    </form>;
  }

  render() {
    const { classes, initialValues, componentLibrary } = this.props;
    const className = classes.root + (
      componentLibrary === BOOTSTRAP ? ' bootstrap-styles' : ''
    )
    return <div className={className}>
      <Formik
        initialValues={initialValues}
        validationSchema={validationSchema}
        onSubmit={this.handleSubmit}
        render={this.renderForm}/>
    </div>
  }
}

function mapStateToProps(state) {
  return {
    componentLibrary: getComponentLibrary(state)
  }
}


export default withStyles(styles)(
    connect(mapStateToProps)(FormikContactForm)
  )

