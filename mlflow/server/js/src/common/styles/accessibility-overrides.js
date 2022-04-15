import { primaryBlue, gray600, green700, orange700 } from './color';

export const accessibilityOverrides = {
  a: {
    // using webapp colors
    color: '#2374bb',

    '&:hover, &:focus': {
      color: '#005580',
    },
  },
  '.staging-tag': {
    color: orange700,
    borderColor: orange700,
    backgroundColor: 'white',
  },
  '.production-tag': {
    color: green700,
    borderColor: green700,
    backgroundColor: 'white',
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
