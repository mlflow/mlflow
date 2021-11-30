import React from 'react';
import { css } from 'emotion';

export function PageContainer({ wut, ...otherProps } = {}) {
  /* eslint-disable prefer-const */
  let rootStyles = styles.basic;
  return <div {...otherProps} className={css(rootStyles)} />;
}

const basicSpacing = 64;

const styles = {
  basic: {
    padding: `0 ${basicSpacing}px`,
    width: '100%',
    flexGrow: 1,
    paddingBottom: 24,
  },
};
