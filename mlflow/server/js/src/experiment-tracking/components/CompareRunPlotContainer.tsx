import type { Theme } from '@emotion/react';

interface CompareRunPlotContainerProps {
  controls: React.ReactNode;
}

export const CompareRunPlotContainer = (props: React.PropsWithChildren<CompareRunPlotContainerProps>) => (
  <div css={styles.wrapper}>
    <div css={styles.controls}>{props.controls}</div>
    <div css={styles.plotWrapper}>{props.children}</div>
  </div>
);

const styles = {
  plotWrapper: {
    overflow: 'hidden',
    width: '100%',
    height: '100%',
    minHeight: 450,
  },
  wrapper: {
    display: 'grid',
    gridTemplateColumns: 'minmax(300px, 1fr) 3fr',
  },
  controls: (theme: Theme) => ({
    padding: `0 ${theme.spacing.xs}px`,
  }),
};
