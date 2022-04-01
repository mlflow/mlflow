import React from 'react';
import { css } from 'emotion';
import { PageWrapper, Spacer } from '@databricks/design-system';

export function PageContainer(props) {
  return (
    <PageWrapper>
      <Spacer />
      <div {...props} className={css(styles.container)} />
    </PageWrapper>
  );
}

const styles = {
  container: {
    width: '100%',
    flexGrow: 1,
    paddingBottom: 24,
  },
};
