/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { useCallback, useRef, useEffect } from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import { SupportsDuBoisThemes } from '@databricks/web-shared/design-system';
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
    <SupportsDuBoisThemes enabled={false}>
      {/* @ts-expect-error TS(2322): Type '() => HTMLElement | undefined' is not assign... Remove this comment to see the full error message */}
      <DesignSystemProvider getPopupContainer={getPopupContainer} isCompact {...props}>
        <ConfigProvider prefixCls='ant'>
          {children}
          {/* @ts-expect-error TS(2322): Type 'MutableRefObject<undefined>' is not assignab... Remove this comment to see the full error message */}
          <div ref={modalContainerElement} />
        </ConfigProvider>
      </DesignSystemProvider>
    </SupportsDuBoisThemes>
  );
};
