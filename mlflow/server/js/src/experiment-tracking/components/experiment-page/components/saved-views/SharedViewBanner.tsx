import type { ReactNode } from 'react';
import { Alert, Button, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/**
 * Read-only banner shown while the user is viewing a shared/saved view. It signals that the view is
 * read-only with respect to local persistence — nothing the user does is saved — and hosts the
 * explicit actions to leave it.
 *
 * Presentational and data-source-agnostic so both the runs and traces tabs can render it:
 * - Runs passes `onOverride` (adopt the shared view into the user's own saved view) + `onDiscard`.
 * - Traces omits `onOverride`, so only the Discard button renders.
 *
 * `componentId` is the base id for the analytics events; the override/discard buttons derive
 * `.override` / `.discard` from it. It is threaded from the consumer (rather than hardcoded) so each
 * tab keeps its own registered componentId namespace.
 */
// Override is all-or-nothing: a consumer that wires up `onOverride` must also supply the button's
// `overrideLabel`, otherwise the override button would render empty. Consumers with no override
// (e.g. traces) omit both. The union enforces this at the type level.
type OverrideProps =
  | { onOverride: () => void; overrideLabel: ReactNode }
  | { onOverride?: never; overrideLabel?: never };

export const SharedViewBanner = ({
  componentId,
  message,
  overrideLabel,
  onOverride,
  onDiscard,
}: {
  componentId: string;
  message: ReactNode;
  onDiscard: () => void;
} & OverrideProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Alert
      componentId={`${componentId}.banner`}
      type="info"
      closable={false}
      css={{ marginBottom: theme.spacing.sm }}
      message={
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: theme.spacing.md,
            flexWrap: 'wrap',
          }}
        >
          <span>{message}</span>
          <div css={{ display: 'flex', gap: theme.spacing.sm, flexShrink: 0 }}>
            {onOverride && (
              <Button componentId={`${componentId}.override`} type="primary" size="small" onClick={onOverride}>
                {overrideLabel}
              </Button>
            )}
            <Button componentId={`${componentId}.discard`} size="small" onClick={onDiscard}>
              <FormattedMessage
                defaultMessage="Discard"
                description="Experiment page > shared view banner > button that discards the shared view and restores the user's own view"
              />
            </Button>
          </div>
        </div>
      }
    />
  );
};
