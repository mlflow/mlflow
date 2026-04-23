/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { PageWrapper, Spacer } from '@databricks/design-system';

type Props = {
  usesFullHeight?: boolean;
  children?: React.ReactNode;
};

export function PageContainer({ usesFullHeight = false, children }: Props) {
  return (
    // @ts-expect-error TS(2322): Type '{ height: string; display: string; flexDirec... Remove this comment to see the full error message
    <PageWrapper css={usesFullHeight ? styles.useFullHeightLayout : styles.wrapper}>
      {/* @ts-expect-error TS(2322): Type '{ css: { flexShrink: number; }; }' is not as... Remove this comment to see the full error message */}
      <Spacer css={styles.fixedSpacer} />
      {usesFullHeight ? children : <div css={styles.container}>{children}</div>}
    </PageWrapper>
  );
}

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
