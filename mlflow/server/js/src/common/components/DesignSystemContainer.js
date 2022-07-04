import React, { useCallback, useRef, useEffect } from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import { message } from 'antd';
import PropTypes from 'prop-types';

const isInsideShadowDOM = (element) =>
  element instanceof window.Node && element.getRootNode() !== document;

/**
 * MFE-safe DesignSystemProvider that checks if the application is
 * in the context of the Shadow DOM and if true, provides dedicated
 * DOM element for the purpose of housing modals/popups there.
 */
export const DesignSystemContainer = (props) => {
  const modalContainerElement = useRef();
  const { children } = props;

  useEffect(() => {
    if (isInsideShadowDOM(modalContainerElement.current)) {
      message.config({
        getContainer: () => modalContainerElement.current,
      });
    }
  }, []);

  const getPopupContainer = useCallback(() => {
    if (isInsideShadowDOM(modalContainerElement.current)) {
      return modalContainerElement.current;
    }
    return document.body;
  }, []);

  return (
    <DesignSystemProvider getPopupContainer={getPopupContainer} {...props}>
      {children}
      <div ref={modalContainerElement} />
    </DesignSystemProvider>
  );
};

DesignSystemContainer.propTypes = {
  children: PropTypes.node.isRequired,
};
