import React from 'react';
import PropTypes from 'prop-types';
import { Radio as AntdRadio } from './antd/Radio';
import { css } from 'emotion';

/**
 * Render a radio button group around ANTD. Styled according to
 * Unified ML design.
 * @param props items: Array of buttons, specified as
 * {value: string, itemContent: node, onClick: (e) => void,
 * dataTestId?: string}.
 * @param props defaultValue: key of the button to select by default.
 */
export class Radio extends React.Component {
  static propTypes = {
    items: PropTypes.arrayOf(PropTypes.shape({
      value: PropTypes.string.isRequired,
      itemContent: PropTypes.node.isRequired,
      onClick: PropTypes.func.isRequired,
      dataTestId: PropTypes.string,
    })),
    defaultValue: PropTypes.string.isRequired,
  }

  render() {
    const { items, defaultValue } = this.props;
    return (
      <div className={css(styles.radioGroup)}>
        <AntdRadio.Group defaultValue={defaultValue} buttonStyle={'solid'} size='large'>
          {items.map(({ value, itemContent, onClick, dataTestId }, i) =>
              <AntdRadio.Button value={value} onClick={onClick} key={i}>
                <div data-test-id={dataTestId}>
                  {itemContent}
                </div>
            </AntdRadio.Button>
          )}
        </AntdRadio.Group>
      </div>
    );
  }
}

const styles = {
  radioGroup: {
    '--text-selected-background-color': 'auto',
    ' .ant-radio-button-wrapper': {
      boxSizing: 'border-box',
      fontSize: '14px',
    },
    /* eslint-disable max-len */
    ' .ant-radio-group-solid .ant-radio-button-wrapper-checked:not(.ant-radio-button-wrapper-disabled)': {
      color: 'var(--primary-text-color, #000000cc)',
      backgroundColor: '#EEEEEE',
      '-webkit-box-shadow': '-1px 0 0 0 #D9D9D9',
      borderColor: '#D9D9D9',
      '::before': {
        'background-color': '#D9D9D9 !important',
      },
    },
  },
};
