import { Button, Modal, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ReactComponent as PromoArrowSvg } from '../../common/static/promo-modal-feature-arrow.svg';
import { FormattedMessage } from 'react-intl';
import { mlflowAliasesLearnMoreLink } from '../../common/constants';

export const ModelsNextUIPromoModal = ({
  visible,
  onClose,
  onTryItNow,
}: {
  visible: boolean;
  onClose: () => void;
  onTryItNow: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Modal
      visible={visible}
      title={
        <FormattedMessage
          defaultMessage='Introducing new way to manage model deployment'
          description='Model registry > OSS Promo modal for model version aliases > modal title'
        />
      }
      onCancel={onClose}
      footer={
        <>
          <Button href={mlflowAliasesLearnMoreLink} rel='noopener' target='_blank'>
            <FormattedMessage
              defaultMessage='Learn more'
              description='Model registry > OSS Promo modal for model version aliases > learn more link'
            />
          </Button>
          <Button type='primary' onClick={onTryItNow}>
            <FormattedMessage
              defaultMessage='Try it now'
              description='Model registry > OSS Promo modal for model version aliases > try it now button label'
            />
          </Button>
        </>
      }
    >
      <div
        css={{
          position: 'relative',
          display: 'grid',
          gridTemplateColumns: '140px 140px 2fr',
          backgroundColor: theme.colors.grey100,
          padding: theme.spacing.lg * 2,
          paddingTop: theme.spacing.lg * 3,
          paddingBottom: theme.spacing.lg * 4,
          alignItems: 'flex-end',
          marginBottom: theme.spacing.md,
          gap: theme.spacing.md,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            alignItems: 'flex-start',
          }}
        >
          <Typography.Title level={3} color='secondary'>
            <FormattedMessage
              defaultMessage='Fixed stages 😞'
              description='Model registry > OSS Promo modal for model version aliases > "before" state showcase'
            />
          </Typography.Title>
          <Tag color='lemon'>Staging</Tag>
          <Tag color='lime'>Production</Tag>
        </div>
        <PromoArrowSvg width='100%' />
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            alignItems: 'flex-start',
          }}
        >
          <Typography.Title level={3} color='secondary'>
            <FormattedMessage
              defaultMessage='Flexible stages 😄'
              description='Model registry > OSS Promo modal for model version aliases > "after" state showcase '
            />
          </Typography.Title>
          <div>
            <Tag color='charcoal'>@ Champion</Tag>
            <Tag>In-review</Tag>
          </div>
          <div>
            <Tag color='charcoal'>@ Challenger</Tag>
            <Tag>Approved</Tag>
          </div>
        </div>
      </div>
      <Typography.Paragraph>
        <FormattedMessage
          defaultMessage={`Introducing Aliases and Tags as a way to manage your model development. Use Aliases e.g.
        @Champion, @Challenger to manage your model lifecycle and tags to help you remember the
        context of each model e.g. @In-Review and @Approved.`}
          description='Model registry > OSS Promo modal for model version aliases > description paragraph body'
        />
      </Typography.Paragraph>
    </Modal>
  );
};
