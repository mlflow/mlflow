import { primaryBlue, gray600 } from './color';

export const accessibilityOverrides = {
  a: {
    // using webapp colors
    color: '#2374bb',

    '&:hover, &:focus': {
      color: '#005580',
    },
  },
  '.ant-btn-primary': {
    borderColor: primaryBlue,
    backgroundColor: primaryBlue,
  },
  '.ant-table-placeholder': {
    color: gray600,
  },
  '.ant-tabs-nav .ant-tabs-tab-active': {
    color: primaryBlue,
  },
  '.ant-radio-button-wrapper-checked': {
    '&:not(.ant-radio-button-wrapper-disabled), &:first-child': {
      color: primaryBlue,
    },
  },
};
