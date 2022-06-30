import React from 'react';
import { DesignSystemProvider } from '@databricks/design-system';
import '@databricks/design-system/dist/index.css';
import { Global } from '@emotion/react';

export const parameters = {
  actions: { argTypesRegex: "^on[A-Z].*" },
  controls: {
    matchers: {
      color: /(background|color)$/i,
      date: /Date$/,
    },
  },
}

export const decorators = [
  (Story) => (
    <DesignSystemProvider isCompact>
      <>
        <Global styles={{
          'html, body': {
            fontSize: 13,
            fontFamily: "-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,Apple Color Emoji,Segoe UI Emoji,Segoe UI Symbol,Noto Color Emoji",
            height: '100%',
          },
          '#root': { height: '100%' },
          '*': {
            boxSizing: 'border-box',
          }
        }} />
        <Story />
      </>
    </DesignSystemProvider>
  )
]
