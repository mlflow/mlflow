/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { PageWrapper, Spacer } from '@databricks/design-system';

type OwnProps = {
  usesFullHeight?: boolean;
  children?: React.ReactNode;
};

// @ts-expect-error TS(2565): Property 'defaultProps' is used before being assig... Remove this comment to see the full error message
type Props = OwnProps & typeof PageContainer.defaultProps;

export function PageContainer(props: Props) {
  const { usesFullHeight, ...restProps } = props;
  return (
    // @ts-expect-error TS(2322): Type '{ height: string; display: string; flexDirec... Remove this comment to see the full error message
    <PageWrapper css={usesFullHeight ? styles.useFullHeightLayout : styles.wrapper}>
      {/* @ts-expect-error TS(2322): Type '{ css: { flexShrink: number; }; }' is not as... Remove this comment to see the full error message */}
      <Spacer css={styles.fixedSpacer} />
      {usesFullHeight ? props.children : <div {...restProps} css={styles.container} />}
    </PageWrapper>
  );
}

PageContainer.defaultProps = {
  usesFullHeight: false,
};

const styles = {
  useFullHeightLayout: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    '&:last-child': {
      flexGrow: 1,
    },
  },
  wrapper: { flex: 1 },
  fixedSpacer: {
    // Ensure spacer's fixed height regardless of flex
    flexShrink: 0,
  },
  container: {
    width: '100%',
    flexGrow: 1,
    paddingBottom: 24,
  },
};
