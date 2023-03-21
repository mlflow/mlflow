import PropTypes from 'prop-types';
import React from 'react';
import { MLFlowRoot } from '../app';
import { MFEAttributesContextProvider } from './MFEAttributesContext';

export function MLFlowMFERoot({ attributes }) {
  return (
    <MFEAttributesContextProvider value={attributes}>
      <MLFlowRoot />
    </MFEAttributesContextProvider>
  );
}

MLFlowMFERoot.propTypes = {
  attributes: PropTypes.object,
};
