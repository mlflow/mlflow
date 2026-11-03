import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { gunzipSync } from 'zlib';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import Utils from '../../common/utils/Utils';
import { setupServer } from '../../common/utils/setup-msw';
import { SkillRegistryApi } from '../api';
import { createMockSkill, createMockSkillVersion } from '../test-utils';
import type { SkillVersion } from '../types';
import { RegisterSkillModal } from './RegisterSkillModal';

const deferred = <T,>() => {
  let resolve: (value: T) => void = () => {};
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
};

const notFound = () => Object.assign(new Error('Skill not found'), { name: 'NotFoundError' });

const folderFile = (path: string, content: string, text?: () => Promise<string>) => {
  const file = new File([content], path.split('/').pop() ?? path);
  const bytes = new TextEncoder().encode(content);
  // jsdom's File has neither text() nor arrayBuffer().
  Object.defineProperties(file, {
    webkitRelativePath: { value: path },
    text: { value: text ?? (async () => content) },
    arrayBuffer: { value: async () => bytes.buffer },
  });
  return file;
};

const readBlob = (blob: Blob) =>
  new Promise<Buffer>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(Buffer.from(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(blob);
  });

const tarNames = (tar: Buffer) => {
  const names: string[] = [];
  for (let offset = 0; offset + 512 <= tar.length; ) {
    const name = tar
      .subarray(offset, offset + 100)
      .toString('utf8')
      .replace(/\0.*$/s, '');
    if (!name) break;
    names.push(name);
    const size = parseInt(
      tar
        .subarray(offset + 124, offset + 136)
        .toString('utf8')
        .replace(/\0.*$/s, '')
        .trim(),
      8,
    );
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  return names;
};

describe('RegisterSkillModal', () => {
  const server = setupServer(
    rest.get(/mlflow\/server-info$/, (_req, res, ctx) =>
      res(ctx.json({ store_type: 'SqlStore', artifact_serving_enabled: true })),
    ),
  );

  const registered = createMockSkillVersion({ name: 'demo', organization: 'acme', version: 1 });
  let onRegistered: jest.Mock<(version: SkillVersion) => void>;
  let onClose: jest.Mock<() => void>;

  beforeEach(() => {
    onRegistered = jest.fn();
    onClose = jest.fn();
    jest.spyOn(SkillRegistryApi, 'getSkill').mockRejectedValue(notFound());
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const renderModal = (skill?: { name: string; organization: string }) =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={new QueryClient()}>
            <RegisterSkillModal visible skill={skill} onClose={onClose} onRegistered={onRegistered} />
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

  const chooseUpload = async () => {
    await userEvent.click(await screen.findByRole('radio', { name: /Upload a folder/ }));
  };

  it('does not hand over a registration that finishes after the dialog unmounted', async () => {
    const registration = deferred<SkillVersion>();
    jest.spyOn(SkillRegistryApi, 'registerSkill').mockReturnValue(registration.promise);
    const { unmount } = renderModal();

    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/main/demo');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(SkillRegistryApi.registerSkill).toHaveBeenCalled());

    // e.g. browser Back while the request is pending.
    unmount();
    await act(async () => {
      registration.resolve(registered);
    });

    expect(onRegistered).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('registers the first version before saving the skill description', async () => {
    const order: string[] = [];
    jest.spyOn(SkillRegistryApi, 'registerSkill').mockImplementation(async () => {
      order.push('register');
      return registered;
    });
    const updateSkill = jest.spyOn(SkillRegistryApi, 'updateSkill').mockImplementation(async () => {
      order.push('update');
      return {} as Awaited<ReturnType<typeof SkillRegistryApi.updateSkill>>;
    });
    const createSkill = jest.spyOn(SkillRegistryApi, 'createSkill');
    renderModal();

    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/main/demo');
    await userEvent.click(screen.getByRole('button', { name: /Advanced settings/ }));
    await userEvent.type(screen.getByLabelText('Description'), 'Reviews code');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(onRegistered).toHaveBeenCalledWith(registered));
    expect(updateSkill).toHaveBeenCalledWith('demo', { description: 'Reviews code' }, 'acme');
    expect(order).toEqual(['register', 'update']);
    expect(createSkill).not.toHaveBeenCalled();
  });

  it('stops when the name is already registered', async () => {
    jest.spyOn(SkillRegistryApi, 'getSkill').mockResolvedValue(createMockSkill({ name: 'demo', organization: 'acme' }));
    const registerSkill = jest.spyOn(SkillRegistryApi, 'registerSkill');
    const updateSkill = jest.spyOn(SkillRegistryApi, 'updateSkill');
    renderModal();

    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/main/demo');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(await screen.findByText('A skill named "@acme/demo" is already registered.')).toBeInTheDocument();
    expect(registerSkill).not.toHaveBeenCalled();
    expect(updateSkill).not.toHaveBeenCalled();
    expect(onRegistered).not.toHaveBeenCalled();
  });

  it('does not delete the skill when the first version request fails', async () => {
    jest.spyOn(SkillRegistryApi, 'registerSkill').mockRejectedValue(new Error('Source unreachable'));
    const deleteSkill = jest.spyOn(SkillRegistryApi, 'deleteSkill').mockResolvedValue({});
    renderModal();

    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/main/demo');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(await screen.findByText('Source unreachable')).toBeInTheDocument();
    expect(deleteSkill).not.toHaveBeenCalled();
    expect(onRegistered).not.toHaveBeenCalled();
  });

  it('keeps the registered version when saving its description fails', async () => {
    jest.spyOn(SkillRegistryApi, 'registerSkill').mockResolvedValue(registered);
    jest.spyOn(SkillRegistryApi, 'updateSkill').mockRejectedValue(new Error('Metadata unavailable'));
    const notify = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    const deleteSkill = jest.spyOn(SkillRegistryApi, 'deleteSkill');
    renderModal();

    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/main/demo');
    await userEvent.click(screen.getByRole('button', { name: /Advanced settings/ }));
    await userEvent.type(screen.getByLabelText('Description'), 'Reviews code');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(onRegistered).toHaveBeenCalledWith(registered));
    expect(notify).toHaveBeenCalledWith(
      'The skill was created, but its description or icons could not be saved: Metadata unavailable',
    );
    expect(deleteSkill).not.toHaveBeenCalled();
  });

  it('says the skill exists when its tags fail to save', async () => {
    jest.spyOn(SkillRegistryApi, 'registerSkill').mockResolvedValue(registered);
    jest.spyOn(SkillRegistryApi, 'setSkillTag').mockRejectedValue(new Error('Tag value too long'));
    const notify = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    renderModal();

    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/main/demo');
    await userEvent.click(screen.getByRole('button', { name: /Advanced settings/ }));
    await userEvent.type(screen.getByLabelText('Key'), 'team');
    await userEvent.type(screen.getByLabelText('Value'), 'platform');
    await userEvent.click(screen.getByRole('button', { name: 'Add tag' }));
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(onRegistered).toHaveBeenCalledWith(registered));
    expect(notify).toHaveBeenCalledWith('The skill was created, but its tags could not be saved: Tag value too long');
  });

  it('keeps a name typed while SKILL.md is still being read', async () => {
    const manifestRead = deferred<string>();
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('demo/SKILL.md', '', () => manifestRead.promise),
    ]);
    await userEvent.type(screen.getByLabelText('Name'), '@acme/typed');
    await act(async () => {
      manifestRead.resolve('---\nname: from-manifest\n---\n');
    });

    expect(screen.getByLabelText('Name')).toHaveValue('@acme/typed');
  });

  it('ignores a SKILL.md read that finishes after switching to Import', async () => {
    const manifestRead = deferred<string>();
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('demo/SKILL.md', '', () => manifestRead.promise),
    ]);
    await userEvent.click(screen.getByRole('radio', { name: /Import from existing source/ }));
    await act(async () => {
      manifestRead.resolve('---\nname: from-manifest\n---\n');
    });

    expect(screen.getByLabelText('Name')).toHaveValue('');
  });

  it("drops the previous folder's name and description when another folder or Import is chosen", async () => {
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('first/SKILL.md', '---\nname: first\ndescription: First skill\n---\n'),
    ]);
    await waitFor(() => expect(screen.getByLabelText('Name')).toHaveValue('first'));
    await userEvent.click(screen.getByRole('button', { name: /Advanced settings/ }));
    expect(screen.getByLabelText('Description')).toHaveValue('First skill');

    // A SKILL.md without frontmatter suggests nothing, so nothing from the first folder may remain.
    await userEvent.upload(screen.getByLabelText('Skill folder'), [folderFile('second/SKILL.md', '# Second\n')]);
    await waitFor(() => expect(screen.getByLabelText('Name')).toHaveValue(''));
    expect(screen.getByLabelText('Description')).toHaveValue('');

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('first/SKILL.md', '---\nname: first\ndescription: First skill\n---\n'),
    ]);
    await waitFor(() => expect(screen.getByLabelText('Name')).toHaveValue('first'));
    await userEvent.click(screen.getByRole('radio', { name: /Import from existing source/ }));
    expect(screen.getByLabelText('Name')).toHaveValue('');
    expect(screen.getByLabelText('Description')).toHaveValue('');
  });

  it('keeps a typed name and description when the folder changes', async () => {
    renderModal();
    await chooseUpload();
    await userEvent.type(screen.getByLabelText('Name'), '@acme/mine');
    await userEvent.click(screen.getByRole('button', { name: /Advanced settings/ }));
    await userEvent.type(screen.getByLabelText('Description'), 'Mine');

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('first/SKILL.md', '---\nname: first\ndescription: First skill\n---\n'),
    ]);

    await waitFor(() => expect(screen.getByLabelText('Name')).toHaveValue('@acme/mine'));
    expect(screen.getByLabelText('Description')).toHaveValue('Mine');
  });

  it('requires a name in the frontmatter of SKILL.md', async () => {
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [folderFile('demo/SKILL.md', '# Demo\n')]);

    expect(await screen.findByText(/SKILL.md needs a name in its frontmatter/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
  });

  it('says when SKILL.md frontmatter cannot be parsed rather than that its name is missing', async () => {
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('demo/SKILL.md', '---\nname: demo\ndescription: [unclosed\n---\n'),
    ]);

    expect(await screen.findByText(/The frontmatter in SKILL.md couldn't be parsed/)).toBeInTheDocument();
    expect(screen.queryByText(/SKILL.md needs a name/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
  });

  it('says when SKILL.md cannot be read', async () => {
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('demo/SKILL.md', '', () => Promise.reject(new Error('NotReadableError'))),
    ]);

    expect(await screen.findByText("Couldn't read SKILL.md. Select the folder again.")).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
  });

  it('refuses a SKILL.md whose name is not a valid skill name', async () => {
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('demo/SKILL.md', '---\nname: My_Skill\n---\n'),
    ]);

    expect(await screen.findByText(/"My_Skill", is not a valid skill name/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
  });

  it('requires a SKILL.md name for a new version too', async () => {
    renderModal({ name: 'demo', organization: 'acme' });
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [folderFile('demo/SKILL.md', '# Demo\n')]);
    expect(await screen.findByText(/SKILL.md needs a name in its frontmatter/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('demo/SKILL.md', '---\nname: demo\n---\n'),
    ]);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create' })).toBeEnabled());
  });

  it('explains a folder without SKILL.md', async () => {
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [folderFile('demo/README.md', '# Demo\n')]);

    expect(
      await screen.findByText('No skills found. Select a folder containing SKILL.md or skill subfolders.'),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
  });

  it('uploads the picked folder and hands over the new version', async () => {
    const registerSkill = jest.spyOn(SkillRegistryApi, 'registerSkill').mockResolvedValue(registered);
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('demo/SKILL.md', '---\nname: demo\n---\n# Demo\n'),
      folderFile('demo/scripts/run.py', 'print(1)\n'),
    ]);
    await waitFor(() => expect(screen.getByLabelText('Name')).toHaveValue('demo'));
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(onRegistered).toHaveBeenCalledWith(registered));
    const [request, content] = registerSkill.mock.calls[0];
    expect(request).toMatchObject({ name: 'demo' });
    expect(content).toBeInstanceOf(Blob);
    expect(tarNames(gunzipSync(await readBlob(content as Blob))).sort()).toEqual(['SKILL.md', 'scripts/run.py']);
  });

  const repositoryFiles = () => [
    folderFile('repo/README.md', '# Repository\n'),
    folderFile('repo/skills/a/SKILL.md', '---\nname: alpha\ndescription: First skill\n---\n'),
    folderFile('repo/skills/a/scripts/run.py', 'print(1)\n'),
    folderFile('repo/skills/b/SKILL.md', '---\nname: beta\n---\n'),
    folderFile('repo/skills/b/reference.md', 'Beta reference\n'),
    folderFile('repo/skills/c/SKILL.md', '---\nname: gamma\n---\n'),
    folderFile('repo/invalid/SKILL.md', '---\nname: Bad_Name\n---\n'),
  ];

  const selectRepositorySkills = async () => {
    await userEvent.upload(screen.getByLabelText('Skill folder'), repositoryFiles());
    await userEvent.click(await screen.findByRole('checkbox', { name: /^alpha/ }));
    await userEvent.click(screen.getByRole('checkbox', { name: /^beta/ }));
  };

  it('registers only selected skills with independent archives and manifest metadata', async () => {
    const registerSkill = jest
      .spyOn(SkillRegistryApi, 'registerSkill')
      .mockImplementation(async (request) =>
        createMockSkillVersion({ name: request.name, organization: request.organization ?? '' }),
      );
    const updateSkill = jest.spyOn(SkillRegistryApi, 'updateSkill').mockResolvedValue(createMockSkill());
    renderModal();
    await chooseUpload();
    await userEvent.upload(screen.getByLabelText('Skill folder'), repositoryFiles());
    expect(await screen.findByRole('checkbox', { name: /^Bad_Name/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
    await userEvent.click(screen.getByRole('checkbox', { name: /^alpha/ }));
    await userEvent.click(screen.getByRole('checkbox', { name: /^beta/ }));
    await userEvent.type(screen.getByLabelText('Organization (optional)'), 'acme');
    await userEvent.click(screen.getByRole('button', { name: 'Create 2 skills' }));
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(onRegistered).not.toHaveBeenCalled();
    expect(registerSkill.mock.calls.map(([request]) => request)).toEqual([
      { name: 'alpha', organization: 'acme', source: null, status: 'active' },
      { name: 'beta', organization: 'acme', source: null, status: 'active' },
    ]);
    expect(updateSkill).toHaveBeenCalledWith('alpha', { description: 'First skill' }, 'acme');
    expect(tarNames(gunzipSync(await readBlob(registerSkill.mock.calls[0][1] as Blob)))).toEqual([
      'SKILL.md',
      'scripts/run.py',
    ]);
    expect(tarNames(gunzipSync(await readBlob(registerSkill.mock.calls[1][1] as Blob)))).toEqual([
      'SKILL.md',
      'reference.md',
    ]);
  });

  it('blocks duplicate manifest names before creating any skills', async () => {
    const registerSkill = jest.spyOn(SkillRegistryApi, 'registerSkill');
    renderModal();
    await chooseUpload();
    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('repo/a/SKILL.md', '---\nname: duplicate\n---\n'),
      folderFile('repo/b/SKILL.md', '---\nname: duplicate\n---\n'),
    ]);
    const choices = await screen.findAllByRole('checkbox');
    await userEvent.click(choices[0]);
    await userEvent.click(choices[1]);
    expect(screen.getByText(/Selected skills have the same name/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create 2 skills' })).toBeDisabled();
    expect(registerSkill).not.toHaveBeenCalled();
  });

  it('packages just the selected skill when creating a version from a parent directory', async () => {
    const createVersion = jest.spyOn(SkillRegistryApi, 'createSkillVersion').mockResolvedValue(registered);
    renderModal({ name: 'demo', organization: 'acme' });
    await chooseUpload();
    await userEvent.upload(screen.getByLabelText('Skill folder'), repositoryFiles());
    await screen.findByText('Choose one skill for this version.');
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
    const select = document.querySelector<HTMLElement>(
      '[data-component-id="mlflow.skill_registry.register_modal.discovered_skill"]',
    );
    if (!select) throw new Error('Discovered skills selector not found');
    await userEvent.click(select);
    await userEvent.click(await screen.findByRole('option', { name: 'beta · skills/b' }));
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(onRegistered).toHaveBeenCalledWith(registered));
    const [name, request, organization, content] = createVersion.mock.calls[0];
    expect({ name, request, organization }).toEqual({
      name: 'demo',
      organization: 'acme',
      request: { source: null, status: 'active' },
    });
    expect(tarNames(gunzipSync(await readBlob(content as Blob)))).toEqual(['SKILL.md', 'reference.md']);
  });

  it('applies upload limits separately to discovered skills before reading manifests', async () => {
    server.use(
      rest.get(/mlflow\/server-info$/, (_req, res, ctx) =>
        res(ctx.json({ store_type: 'SqlStore', artifact_serving_enabled: true, skill_content_max_files: 1 })),
      ),
    );
    const readLargeManifest = jest.fn<() => Promise<string>>();
    renderModal();
    await chooseUpload();
    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('repo/a/SKILL.md', '---\nname: alpha\n---\n', readLargeManifest),
      folderFile('repo/a/extra.md', 'Too many files'),
      folderFile('repo/b/SKILL.md', '---\nname: beta\n---\n'),
    ]);
    expect(await screen.findByRole('checkbox', { name: /^a a/ })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: /^beta/ })).toBeEnabled();
    expect(readLargeManifest).not.toHaveBeenCalled();
  });

  it('ignores discovery from an older folder selection', async () => {
    const oldRead = deferred<string>();
    renderModal();
    await chooseUpload();
    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('repo/old/SKILL.md', '', () => oldRead.promise),
    ]);
    await userEvent.upload(screen.getByLabelText('Skill folder'), [
      folderFile('repo/new/SKILL.md', '---\nname: latest\n---\n'),
    ]);
    await waitFor(() => expect(screen.getByLabelText('Name')).toHaveValue('latest'));
    await act(async () => oldRead.resolve('---\nname: outdated\n---\n'));
    expect(screen.getByLabelText('Name')).toHaveValue('latest');
    expect(screen.queryByText('Finding skills…')).not.toBeInTheDocument();
  });

  it('checks all names before creating a batch', async () => {
    jest.spyOn(SkillRegistryApi, 'getSkill').mockImplementation(async (name) => {
      if (name === 'beta') return createMockSkill({ name });
      throw notFound();
    });
    const registerSkill = jest.spyOn(SkillRegistryApi, 'registerSkill');
    renderModal();
    await chooseUpload();
    await selectRepositorySkills();
    await userEvent.click(screen.getByRole('button', { name: 'Create 2 skills' }));
    expect(await screen.findByText('A skill named "beta" is already registered.')).toBeInTheDocument();
    expect(registerSkill).not.toHaveBeenCalled();
  });

  it('retries a partial batch without registering completed skills again', async () => {
    let failed = false;
    const registerSkill = jest.spyOn(SkillRegistryApi, 'registerSkill').mockImplementation(async (request) => {
      if (request.name === 'beta' && !failed) {
        failed = true;
        throw new Error('Upload failed');
      }
      return createMockSkillVersion({ name: request.name });
    });
    jest.spyOn(SkillRegistryApi, 'updateSkill').mockResolvedValue(createMockSkill());
    renderModal();
    await chooseUpload();
    await selectRepositorySkills();
    await userEvent.click(screen.getByRole('button', { name: 'Create 2 skills' }));
    expect(await screen.findByText('Upload failed')).toBeInTheDocument();
    expect(screen.getByText('1 of 2 skills registered. Retry skips the completed skills.')).toBeInTheDocument();
    expect(screen.getByLabelText('Skill folder')).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Retry remaining' }));
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(registerSkill.mock.calls.map(([request]) => request.name)).toEqual(['alpha', 'beta', 'beta']);
  });

  it('stops the batch when closed during the first registration', async () => {
    const firstRegistration = deferred<SkillVersion>();
    const registerSkill = jest.spyOn(SkillRegistryApi, 'registerSkill').mockReturnValue(firstRegistration.promise);
    jest.spyOn(SkillRegistryApi, 'updateSkill').mockResolvedValue(createMockSkill());
    renderModal();
    await chooseUpload();
    await selectRepositorySkills();
    await userEvent.click(screen.getByRole('button', { name: 'Create 2 skills' }));
    await waitFor(() => expect(registerSkill).toHaveBeenCalledTimes(1));
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    await act(async () => firstRegistration.resolve(createMockSkillVersion({ name: 'alpha' })));
    expect(registerSkill).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onRegistered).not.toHaveBeenCalled();
  });
});
