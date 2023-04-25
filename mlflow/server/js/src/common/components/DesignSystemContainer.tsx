import React, { useCallback, useRef, useEffect } from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import { message, ConfigProvider } from 'antd';

const isInsideShadowDOM = (element: any) =>
  element instanceof window.Node && element.getRootNode() !== document;

type DesignSystemContainerProps = {
  children: React.ReactNode;
};

/**
 * MFE-safe DesignSystemProvider that checks if the application is
 * in the context of the Shadow DOM and if true, provides dedicated
 * DOM element for the purpose of housing modals/popups there.
 */
export const DesignSystemContainer = (props: DesignSystemContainerProps) => {
  const modalContainerElement = useRef();
  const { children } = props;

  useEffect(() => {
    if (isInsideShadowDOM(modalContainerElement.current)) {
      message.config({
        // @ts-expect-error TS(2322): Type '() => undefined' is not assignable to type '... Remove this comment to see the full error message
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
    // @ts-expect-error TS(2322): Type '() => HTMLElement | undefined' is not assign... Remove this comment to see the full error message
    <DesignSystemProvider getPopupContainer={getPopupContainer} isCompact {...props}>
      <ConfigProvider prefixCls='ant'>
        {children}
        {/* @ts-expect-error TS(2322): Type 'MutableRefObject<undefined>' is not assignab... Remove this comment to see the full error message */}
        <div ref={modalContainerElement} />
      </ConfigProvider>
    </DesignSystemProvider>
  );
};
