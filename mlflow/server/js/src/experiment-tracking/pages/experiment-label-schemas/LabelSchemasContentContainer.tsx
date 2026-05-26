import { useState } from 'react';
import { Button, Empty, PlusIcon, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas/hooks/useListLabelSchemasQuery';
import type { LabelSchema } from '../../components/label-schemas/types';
import { DeleteLabelSchemaModal } from './DeleteLabelSchemaModal';
import { LabelSchemaCard } from './LabelSchemaCard';
import { LabelSchemaModal } from './LabelSchemaModal';

export interface LabelSchemasContentContainerProps {
  experimentId: string;
}

export const LabelSchemasContentContainer = ({ experimentId }: LabelSchemasContentContainerProps) => {
  const { theme } = useDesignSystemTheme();
  const [modalVisible, setModalVisible] = useState(false);
  const [editingSchema, setEditingSchema] = useState<LabelSchema | null>(null);
  const [deletingSchema, setDeletingSchema] = useState<LabelSchema | null>(null);

  const { labelSchemas, nextPageToken, isLoading, error } = useListLabelSchemasQuery({
    experimentId,
  });

  const openCreateModal = () => {
    setEditingSchema(null);
    setModalVisible(true);
  };

  const openEditModal = (schema: LabelSchema) => {
    setEditingSchema(schema);
    setModalVisible(true);
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

  return (
    <div css={{ padding: theme.spacing.lg, display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
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
              onEdit={openEditModal}
              onDelete={setDeletingSchema}
            />
          ))}
          {nextPageToken && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Showing the first {count} schemas. Additional schemas exist for this experiment but are not displayed; pagination support is coming."
                description="Label schemas list truncation notice"
                values={{ count: labelSchemas.length }}
              />
            </Typography.Text>
          )}
        </div>
      )}

      <LabelSchemaModal
        experimentId={experimentId}
        editingSchema={editingSchema}
        visible={modalVisible}
        onClose={closeModal}
      />
      <DeleteLabelSchemaModal schema={deletingSchema} onClose={() => setDeletingSchema(null)} />
    </div>
  );
};
