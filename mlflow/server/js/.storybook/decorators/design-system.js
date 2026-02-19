import React from 'react';
import { Global } from '@emotion/react';
import { useRef } from 'react';
import { DesignSystemContainer } from '../../src/common/components/DesignSystemContainer';

export const designSystemDecorator = (Story) => {
  const modalContainerRef = useRef(null);

  return (
    <DesignSystemContainer isCompact getPopupContainer={() => modalContainerRef.current}>
      <>
        <Global
          styles={{
            'html, body': {
              fontSize: 13,
              fontFamily:
                '-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,Apple Color Emoji,Segoe UI Emoji,Segoe UI Symbol,Noto Color Emoji',
              height: '100%',
            },
            '#root': { height: '100%' },
            '*': {
              boxSizing: 'border-box',
            },
          }}
        />
        <Story />
        <div ref={modalContainerRef} />
      </>
    </DesignSystemContainer>
  );
};
