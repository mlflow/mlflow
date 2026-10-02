import { useState } from 'react';
import { Alert, FormUI, Input, Modal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { SkillRegistryApi } from '../api';
import { SkillIconEditor } from '../components/SkillIconEditor';
import type { RegistryIcon, Skill, UpdateSkillRequest } from '../types';
import { useInvalidateSkillQueries } from './useInvalidateSkillQueries';

export const useEditSkillModal = ({ name, organization }: { name: string; organization: string }) => {
  const { theme } = useDesignSystemTheme();
  const invalidate = useInvalidateSkillQueries();
  const [visible, setVisible] = useState(false);
  const [description, setDescription] = useState('');
  const [icons, setIcons] = useState<RegistryIcon[]>([]);

  const mutation = useMutation<unknown, Error, UpdateSkillRequest>({
    mutationFn: (request) => SkillRegistryApi.updateSkill(name, request, organization),
    onSuccess: () => invalidate(name, organization),
  });

  const openEditSkill = (skill: Skill) => {
    setDescription(skill.description || '');
    setIcons(skill.icons ?? []);
    mutation.reset();
    setVisible(true);
  };

  const handleSave = () => {
    const savedIcons = icons.filter((icon) => icon.src.trim());
    mutation.mutate(
      {
        description: description.trim() || null,
        icons: savedIcons.length ? savedIcons : null,
      },
      { onSuccess: () => setVisible(false) },
    );
  };

  const EditSkillModal = visible ? (
    <Modal
      componentId="mlflow.skill_registry.edit_skill_modal"
      title={
        <FormattedMessage defaultMessage="Edit skill" description="Title for editing a skill's description and icons" />
      }
      visible={visible}
      confirmLoading={mutation.isLoading}
      okText={<FormattedMessage defaultMessage="Save" description="Save skill presentation edits" />}
      onOk={handleSave}
      onCancel={() => {
        mutation.reset();
        setVisible(false);
      }}
    >
      {mutation.error && (
        <Alert
          componentId="mlflow.skill_registry.edit_skill_modal.error"
          type="error"
          closable={false}
          message={mutation.error.message}
          css={{ marginBottom: theme.spacing.md }}
        />
      )}
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <FormUI.Label htmlFor="mlflow.skill_registry.edit_skill_modal.description">
            <FormattedMessage defaultMessage="Description" description="Label for the skill description editor" />
          </FormUI.Label>
          <Input.TextArea
            id="mlflow.skill_registry.edit_skill_modal.description"
            componentId="mlflow.skill_registry.edit_skill_modal.description"
            value={description}
            rows={3}
            onChange={(event) => setDescription(event.target.value)}
          />
        </div>
        <SkillIconEditor icons={icons} onChange={setIcons} />
      </div>
    </Modal>
  ) : null;

  return { EditSkillModal, openEditSkill };
};
