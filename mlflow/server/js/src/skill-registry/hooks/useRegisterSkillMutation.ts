import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { SkillRegistryApi } from '../api';
import type {
  ExternalSkillVersionRequest,
  RegisterExternalSkillRequest,
  RegisterUploadedSkillRequest,
  SkillVersion,
  UploadedSkillVersionRequest,
} from '../types';

export type RegisterSkillMutationInput =
  | { kind: 'register'; request: RegisterExternalSkillRequest }
  | { kind: 'register-upload'; request: RegisterUploadedSkillRequest; content: Blob }
  | { kind: 'version'; name: string; organization: string; request: ExternalSkillVersionRequest }
  | { kind: 'version-upload'; name: string; organization: string; request: UploadedSkillVersionRequest; content: Blob };

// The caller refreshes the skill queries once its follow-up writes are done.
export const useRegisterSkillMutation = () =>
  useMutation<SkillVersion, Error, RegisterSkillMutationInput>({
    mutationFn: (input) => {
      if (input.kind === 'register') {
        return SkillRegistryApi.registerSkill(input.request);
      }
      if (input.kind === 'register-upload') {
        return SkillRegistryApi.registerSkill(input.request, input.content);
      }
      if (input.kind === 'version-upload') {
        return SkillRegistryApi.createSkillVersion(input.name, input.request, input.organization, input.content);
      }
      return SkillRegistryApi.createSkillVersion(input.name, input.request, input.organization);
    },
  });
