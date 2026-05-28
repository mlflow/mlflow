import { useMemo, useState } from 'react';
import { Button, Empty, PlusIcon, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { ModelTraceExplorerResizablePane } from '@databricks/web-shared/model-trace-explorer';

import { useListLabelSchemasQuery } from '../../components/label-schemas/hooks/useListLabelSchemasQuery';
import type { LabelSchema } from '../../components/label-schemas/types';
import { DeleteLabelSchemaModal } from './DeleteLabelSchemaModal';
import { LabelSchemaCard } from './LabelSchemaCard';
import { LabelSchemaModal } from './LabelSchemaModal';
import { LabelSchemaPreview } from './LabelSchemaPreview';
import { getFormValuesFromSchema, type LabelSchemaFormData } from './labelSchemaFormUtils';

export interface LabelSchemasContentContainerProps {
  experimentId: string;
}

export const LabelSchemasContentContainer = ({ experimentId }: LabelSchemasContentContainerProps) => {
  const { theme } = useDesignSystemTheme();
  const [modalVisible, setModalVisible] = useState(false);
  const [editingSchema, setEditingSchema] = useState<LabelSchema | null>(null);
  const [deletingSchema, setDeletingSchema] = useState<LabelSchema | null>(null);
  const [selectedSchemaId, setSelectedSchemaId] = useState<string | null>(null);
  // Live form state piped from the modal so the preview pane updates as
  // the user types. `null` when the modal is closed.
  const [liveFormData, setLiveFormData] = useState<LabelSchemaFormData | null>(null);
  // Track the LEFT pane width across renders for `ModelTraceExplorerResizablePane`.
  // The pane's own `useLayoutEffect` overrides this on first measurement
  // with `containerWidth * initialRatio`, so the value here is only used
  // for the brief moment before the layout effect fires; we use 0 to
  // avoid a misleading default that doesn't match the resting layout.
  const [leftPaneWidth, setLeftPaneWidth] = useState(0);

  const { labelSchemas, nextPageToken, isLoading, error } = useListLabelSchemasQuery({
    experimentId,
  });

  // Default the selection to the first schema once data loads. `useMemo`
  // keeps the chosen schema stable across re-renders (the list query
  // returns a fresh array reference each fetch).
  const selectedSchema = useMemo<LabelSchema | null>(() => {
    if (labelSchemas.length === 0) {
      return null;
    }
    if (selectedSchemaId) {
      const match = labelSchemas.find((s) => s.schema_id === selectedSchemaId);
      if (match) {
        return match;
      }
    }
    return labelSchemas[0];
  }, [labelSchemas, selectedSchemaId]);

  // Preview source: live form state (modal open) takes precedence;
  // otherwise the currently-selected card's saved state, converted to
  // form-data shape so the preview always renders from a single
  // canonical input type.
  const previewFormData = useMemo<LabelSchemaFormData | null>(() => {
    if (liveFormData) {
      return liveFormData;
    }
    if (selectedSchema) {
      return getFormValuesFromSchema(selectedSchema);
    }
    return null;
  }, [liveFormData, selectedSchema]);

  const openCreateModal = () => {
    setEditingSchema(null);
    setModalVisible(true);
  };

  const openEditModal = (schema: LabelSchema) => {
    setEditingSchema(schema);
    setModalVisible(true);
    // Keep the highlighted card and the preview's source schema in
    // sync while editing: otherwise the card list would still
    // highlight whichever schema was selected before Edit was
    // clicked, even though the modal now drives the preview pane.
    setSelectedSchemaId(schema.schema_id);
  };

  const closeModal = () => {
    setModalVisible(false);
    setEditingSchema(null);
  };

  if (isLoading) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: 400,
        }}
      >
        <Spinner size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: 400,
        }}
      >
        <Empty
          title={
            <FormattedMessage
              defaultMessage="Unable to load label schemas"
              description="Label schemas list error title"
            />
          }
          description={<span>{error.message}</span>}
        />
      </div>
    );
  }

  const leftColumn = (
    <div
      css={{
        padding: theme.spacing.lg,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        height: '100%',
        overflowY: 'auto',
      }}
    >
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Typography.Title level={3} withoutMargins>
          <FormattedMessage defaultMessage="Labeling schemas" description="Labeling schemas page heading" />
        </Typography.Title>
        <Button
          componentId="mlflow.experiment-label-schemas.create-button"
          icon={<PlusIcon />}
          type="primary"
          onClick={openCreateModal}
        >
          <FormattedMessage defaultMessage="New schema" description="Create new schema button" />
        </Button>
      </div>

      {labelSchemas.length === 0 ? (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            minHeight: 400,
            width: '100%',
            '& > div': {
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
            },
          }}
        >
          <Empty
            title={
              <FormattedMessage defaultMessage="No label schemas yet" description="Label schemas empty state title" />
            }
            description={
              <FormattedMessage
                defaultMessage="Create a schema to define how SMEs label traces in the review UI: pass/fail, categorical, or numeric."
                description="Label schemas empty state description"
              />
            }
          />
        </div>
      ) : (
        <div>
          {labelSchemas.map((schema) => (
            <LabelSchemaCard
              key={schema.schema_id}
              schema={schema}
              selected={selectedSchema?.schema_id === schema.schema_id}
              onSelect={(s) => setSelectedSchemaId(s.schema_id)}
              onEdit={openEditModal}
              onDelete={setDeletingSchema}
            />
          ))}
          {nextPageToken && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Additional schemas exist for this experiment but are not displayed. Pagination support is coming; the first {count} are shown above."
                description="Label schemas list truncation notice"
                values={{ count: labelSchemas.length }}
              />
            </Typography.Text>
          )}
        </div>
      )}
    </div>
  );

  const rightColumn = (
    <div
      css={{
        height: '100%',
        borderLeft: `1px solid ${theme.colors.border}`,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
      }}
    >
      <LabelSchemaPreview formData={previewFormData} />
    </div>
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div css={{ flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden' }}>
        <ModelTraceExplorerResizablePane
          initialRatio={0.55}
          paneWidth={leftPaneWidth}
          setPaneWidth={setLeftPaneWidth}
          leftMinWidth={320}
          rightMinWidth={320}
          leftChild={leftColumn}
          rightChild={rightColumn}
        />
      </div>
      <LabelSchemaModal
        experimentId={experimentId}
        editingSchema={editingSchema}
        visible={modalVisible}
        onClose={closeModal}
        onFormDataChange={setLiveFormData}
        onCreateSuccess={setSelectedSchemaId}
      />
      <DeleteLabelSchemaModal schema={deletingSchema} onClose={() => setDeletingSchema(null)} />
    </div>
  );
};
