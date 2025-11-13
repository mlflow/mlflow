import { css } from '@emotion/react';
export const getDayPickerStyles = (prefix, theme) => css `
  .${prefix} {
    --rdp-cell-size: ${theme.general.heightSm}px;
    --rdp-caption-font-size: ${theme.typography.fontSizeBase}px;
    --rdp-accent-color: ${theme.colors.actionPrimaryBackgroundDefault};
    --rdp-background-color: ${theme.colors.actionTertiaryBackgroundPress};
    --rdp-accent-color-dark: ${theme.colors.actionPrimaryBackgroundDefault};
    --rdp-background-color-dark: ${theme.colors.actionTertiaryBackgroundPress};
    --rdp-outline: 2px solid var(--rdp-accent-color);
    --rdp-outline-selected: 3px solid var(--rdp-accent-color);
    --rdp-selected-color: #fff;
    padding: 4px;
  }

  .${prefix}-vhidden {
    box-sizing: border-box;
    padding: 0;
    margin: 0;
    background: transparent;
    border: 0;
    -moz-appearance: none;
    -webkit-appearance: none;
    appearance: none;
    position: absolute !important;
    top: 0;
    width: 1px !important;
    height: 1px !important;
    padding: 0 !important;
    overflow: hidden !important;
    clip: rect(1px, 1px, 1px, 1px) !important;
    border: 0 !important;
  }

  .${prefix}-button_reset {
    appearance: none;
    position: relative;
    margin: 0;
    padding: 0;
    cursor: default;
    color: inherit;
    background: none;
    font: inherit;
    -moz-appearance: none;
    -webkit-appearance: none;
  }

  .${prefix}-button_reset:focus-visible {
    outline: none;
  }

  .${prefix}-button {
    border: 2px solid transparent;
  }

  .${prefix}-button[disabled]:not(.${prefix}-day_selected) {
    opacity: 0.25;
  }

  .${prefix}-button:not([disabled]) {
    cursor: pointer;
  }

  .${prefix}-button:focus-visible:not([disabled]) {
    color: inherit;
    background-color: var(--rdp-background-color);
    border: var(--rdp-outline);
  }

  .${prefix}-button:hover:not([disabled]):not(.${prefix}-day_selected) {
    background-color: var(--rdp-background-color);
  }

  .${prefix}-months {
    display: flex;
    justify-content: center;
  }

  .${prefix}-month {
    margin: 0 1em;
  }

  .${prefix}-month:first-of-type {
    margin-left: 0;
  }

  .${prefix}-month:last-child {
    margin-right: 0;
  }

  .${prefix}-table {
    margin: 0;
    max-width: calc(var(--rdp-cell-size) * 7);
    border-collapse: collapse;
  }

  .${prefix}-with_weeknumber .${prefix}-table {
    max-width: calc(var(--rdp-cell-size) * 8);
    border-collapse: collapse;
  }

  .${prefix}-caption {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0;
    text-align: left;
  }

  .${prefix}-multiple_months .${prefix}-caption {
    position: relative;
    display: block;
    text-align: center;
  }

  .${prefix}-caption_dropdowns {
    position: relative;
    display: inline-flex;
  }

  .${prefix}-caption_label {
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    margin: 0;
    padding: 0 0.25em;
    white-space: nowrap;
    color: currentColor;
    border: 0;
    border: 2px solid transparent;
    font-family: inherit;
    font-size: var(--rdp-caption-font-size);
    font-weight: 600;
  }

  .${prefix}-nav {
    white-space: nowrap;
  }

  .${prefix}-multiple_months .${prefix}-caption_start .${prefix}-nav {
    position: absolute;
    top: 50%;
    left: 0;
    transform: translateY(-50%);
  }

  .${prefix}-multiple_months .${prefix}-caption_end .${prefix}-nav {
    position: absolute;
    top: 50%;
    right: 0;
    transform: translateY(-50%);
  }

  .${prefix}-nav_button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: var(--rdp-cell-size);
    height: var(--rdp-cell-size);
  }

  .${prefix}-dropdown_year, .${prefix}-dropdown_month {
    position: relative;
    display: inline-flex;
    align-items: center;
  }

  .${prefix}-dropdown {
    appearance: none;
    position: absolute;
    z-index: 2;
    top: 0;
    bottom: 0;
    left: 0;
    width: 100%;
    margin: 0;
    padding: 0;
    cursor: inherit;
    opacity: 0;
    border: none;
    background-color: transparent;
    font-family: inherit;
    font-size: inherit;
    line-height: inherit;
  }

  .${prefix}-dropdown[disabled] {
    opacity: unset;
    color: unset;
  }

  .${prefix}-dropdown:focus-visible:not([disabled]) + .${prefix}-caption_label {
    background-color: var(--rdp-background-color);
    border: var(--rdp-outline);
    border-radius: 6px;
  }

  .${prefix}-dropdown_icon {
    margin: 0 0 0 5px;
  }

  .${prefix}-head {
    border: 0;
  }

  .${prefix}-head_row, .${prefix}-row {
    height: 100%;
  }

  .${prefix}-head_cell {
    vertical-align: middle;
    font-size: inherit;
    font-weight: 400;
    color: ${theme.colors.textSecondary};
    text-align: center;
    height: 100%;
    height: var(--rdp-cell-size);
    padding: 0;
    text-transform: uppercase;
  }

  .${prefix}-tbody {
    border: 0;
  }

  .${prefix}-tfoot {
    margin: 0.5em;
  }

  .${prefix}-cell {
    width: var(--rdp-cell-size);
    height: 100%;
    height: var(--rdp-cell-size);
    padding: 0;
    text-align: center;
  }

  .${prefix}-weeknumber {
    font-size: 0.75em;
  }

  .${prefix}-weeknumber, .${prefix}-day {
    display: flex;
    overflow: hidden;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    width: var(--rdp-cell-size);
    max-width: var(--rdp-cell-size);
    height: var(--rdp-cell-size);
    margin: 0;
    border: 2px solid transparent;
    border-radius: ${theme.general.borderRadiusBase}px;
  }

  .${prefix}-day_today:not(.${prefix}-day_outside) {
    font-weight: bold;
  }

  .${prefix}-day_selected, .${prefix}-day_selected:focus-visible, .${prefix}-day_selected:hover {
    color: var(--rdp-selected-color);
    opacity: 1;
    background-color: var(--rdp-accent-color);
  }

  .${prefix}-day_outside {
    pointer-events: none;
    color: ${theme.colors.textSecondary};
  }

  .${prefix}-day_selected:focus-visible {
    outline: var(--rdp-outline);
    outline-offset: 2px;
    z-index: 1;
  }

  .${prefix}:not([dir='rtl']) .${prefix}-day_range_start:not(.${prefix}-day_range_end) {
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
  }

  .${prefix}:not([dir='rtl']) .${prefix}-day_range_end:not(.${prefix}-day_range_start) {
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
  }

  .${prefix}[dir='rtl'] .${prefix}-day_range_start:not(.${prefix}-day_range_end) {
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
  }

  .${prefix}[dir='rtl'] .${prefix}-day_range_end:not(.${prefix}-day_range_start) {
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
  }

  .${prefix}-day_range_start, .${prefix}-day_range_end {
    border: 0;
    & > span {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: ${theme.general.borderRadiusBase}px;
      background-color: var(--rdp-accent-color);
      color: ${theme.colors.white};
    }
  }

  .${prefix}-day_range_end.${prefix}-day_range_start {
    border-radius: ${theme.general.borderRadiusBase}px;
  }

  .${prefix}-day_range_middle {
    border-radius: 0;
    background-color: var(--rdp-background-color);
    color: ${theme.colors.actionDefaultTextDefault};

    &:hover {
      color: ${theme.colors.actionTertiaryTextHover};
    }
  }

  .${prefix}-row > td:last-of-type .${prefix}-day_range_middle {
    border-top-right-radius: ${theme.general.borderRadiusBase}px;
    border-bottom-right-radius: ${theme.general.borderRadiusBase}px;
  }

  .${prefix}-row > td:first-of-type .${prefix}-day_range_middle {
    border-top-left-radius: ${theme.general.borderRadiusBase}px;
    border-bottom-left-radius: ${theme.general.borderRadiusBase}px;
  }
`;
//# sourceMappingURL=styles.js.map