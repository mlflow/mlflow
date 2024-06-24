import { Button, CloseIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { Global } from '@emotion/react';
import { FormattedJsonDisplay } from 'common/components/JsonFormatting';
import { isUndefined } from 'lodash';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { ResizableBox } from 'react-resizable';

const initialWidth = 200;
const maxWidth = 500;

export const ShowArtifactLoggedTableViewDataPreview = ({
  data,
  onClose,
}: {
  data: string | undefined;
  onClose: VoidFunction;
}) => {
  const { theme } = useDesignSystemTheme();
  const [dragging, setDragging] = useState(false);

  if (isUndefined(data)) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        height: '100%',
        flexDirection: 'row-reverse',
        position: 'relative',
        borderLeft: `1px solid ${theme.colors.border}`,
      }}
    >
      {dragging && (
        <Global
          styles={{
            'body, :host': {
              userSelect: 'none',
            },
          }}
        />
      )}
      <ResizableBox
        width={initialWidth}
        height={undefined}
        axis="x"
        resizeHandles={['w']}
        minConstraints={[initialWidth, 150]}
        maxConstraints={[maxWidth, 150]}
        onResizeStart={() => setDragging(true)}
        onResizeStop={() => setDragging(false)}
        handle={
          <div
            css={{
              width: theme.spacing.xs,
              left: -(theme.spacing.xs / 2),
              height: '100%',
              position: 'absolute',
              top: 0,
              cursor: 'ew-resize',
              '&:hover': {
                backgroundColor: theme.colors.border,
                opacity: 0.5,
              },
            }}
          />
        }
        css={{
          position: 'relative',
        }}
      >
        <div css={{ padding: theme.spacing.sm }}>
          <div css={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography.Title level={5}>
              <FormattedMessage
                defaultMessage="Preview"
                description="Run page > artifact view > logged table view > preview box > title"
              />
            </Typography.Title>
            <Button
              componentId="mlflow.run.artifact_view.preview_close"
              onClick={() => onClose()}
              icon={<CloseIcon />}
            />
          </div>
          {!data && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Click a cell to preview data"
                description="Run page > artifact view > logged table view > preview box > CTA"
              />
            </Typography.Text>
          )}
          <FormattedJsonDisplay json={data} />
        </div>
      </ResizableBox>
    </div>
  );
};
