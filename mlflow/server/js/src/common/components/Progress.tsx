import type { Theme } from '@emotion/react';

export interface ProgressProps {
  percent: number;
  format: (percent: number) => string;
  className?: string;
}

/**
 * Recreates basic features of antd's <Progress /> component.
 * Temporary solution, waiting for this component to be included in DuBois.
 */
export const Progress = (props: ProgressProps) => {
  return (
    <div css={styles.wrapper} className={props.className}>
      <div css={styles.track}>
        <div css={styles.progressTrack} style={{ width: `${props.percent}%` }} />
      </div>
      {props.format(props.percent)}
    </div>
  );
};

const styles = {
  wrapper: (theme: Theme) => ({ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }),
  track: (theme: Theme) => ({
    backgroundColor: theme.colors.backgroundSecondary,
    height: theme.spacing.sm,
    flex: 1,
    borderRadius: theme.spacing.sm,
  }),
  progressTrack: (theme: Theme) => ({
    backgroundColor: theme.colors.primary,
    height: theme.spacing.sm,
    borderRadius: theme.spacing.sm,
  }),
};
