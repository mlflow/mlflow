import React from 'react';
import PropTypes from 'prop-types';
import { PageWrapper, Spacer } from '@databricks/design-system';

export function PageContainer(props) {
  return (
    <PageWrapper css={props.usesFullHeight ? styles.useFullHeightLayout : styles.wrapper}>
      <Spacer />
      {props.usesFullHeight ? props.children : <div {...props} css={styles.container} />}
    </PageWrapper>
  );
}

PageContainer.propTypes = {
  usesFullHeight: PropTypes.bool,
  children: PropTypes.node,
};

PageContainer.defaultProps = {
  usesFullHeight: false,
};

const styles = {
  useFullHeightLayout: { height: '100%', display: 'grid', gridTemplateRows: 'auto auto 1fr' },
  wrapper: { flex: 1 },
  container: {
    width: '100%',
    flexGrow: 1,
    paddingBottom: 24,
  },
};
