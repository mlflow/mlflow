import { Global } from '@emotion/react';

export const gray600 = '#6B6B6B';

export const blue500 = '#2374BB';

export const accessibilityOverrides = {
  a: {
    // using webapp colors
    color: '#2374bb',

    '&:hover, &:focus': {
      color: '#005580',
    },
  },
  '.ant-btn-primary': {
    borderColor: blue500,
    backgroundColor: blue500,
  },
  '.ant-table-placeholder': {
    color: gray600,
  },
  '.ant-tabs-nav .ant-tabs-tab-active': {
    color: blue500,
  },
  '.ant-radio-button-wrapper-checked': {
    '&:not(.ant-radio-button-wrapper-disabled), &:first-child': {
      color: blue500,
    },
  },
};

export const AccessibilityOverridesStyles = () => <Global styles={accessibilityOverrides} />;
