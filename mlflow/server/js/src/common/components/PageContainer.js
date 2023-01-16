import React from 'react';
import { PageWrapper, Spacer } from '@databricks/design-system';

export function PageContainer(props) {
  return (
    <PageWrapper css={styles.wrapper}>
      <Spacer />
      <div {...props} css={styles.container} />
    </PageWrapper>
  );
}

const styles = {
  wrapper: { flex: 1 },
  container: {
    width: '100%',
    flexGrow: 1,
    paddingBottom: 24,
  },
};
