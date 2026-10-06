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
  setupServer(
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
    jest.spyOn(SkillRegistryApi, 'createSkill').mockImplementation(async (request) => createMockSkill(request));
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const renderModal = () =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={new QueryClient()}>
            <RegisterSkillModal visible onClose={onClose} onRegistered={onRegistered} />
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

  it('creates a new skill with its description before registering its first version', async () => {
    const order: string[] = [];
    const createSkill = jest.spyOn(SkillRegistryApi, 'createSkill').mockImplementation(async (request) => {
      order.push('create');
      return createMockSkill(request);
    });
    jest.spyOn(SkillRegistryApi, 'registerSkill').mockImplementation(async () => {
      order.push('register');
      return registered;
    });
    renderModal();

    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/main/demo');
    await userEvent.click(screen.getByRole('button', { name: /Advanced settings/ }));
    await userEvent.type(screen.getByLabelText('Description'), 'Reviews code');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(onRegistered).toHaveBeenCalledWith(registered));
    expect(createSkill).toHaveBeenCalledWith({ name: 'demo', organization: 'acme', description: 'Reviews code' });
    expect(order).toEqual(['create', 'register']);
  });

  it('stops when the name was taken before the skill could be created', async () => {
    const registerSkill = jest.spyOn(SkillRegistryApi, 'registerSkill');
    jest.spyOn(SkillRegistryApi, 'createSkill').mockRejectedValue(new Error('Skill already exists'));
    // Free when the dialog checked, taken by the time it creates the skill.
    jest
      .spyOn(SkillRegistryApi, 'getSkill')
      .mockRejectedValueOnce(notFound())
      .mockResolvedValue(createMockSkill({ name: 'demo', organization: 'acme' }));
    renderModal();

    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/main/demo');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(await screen.findByText('A skill named "@acme/demo" is already registered.')).toBeInTheDocument();
    expect(registerSkill).not.toHaveBeenCalled();
    expect(onRegistered).not.toHaveBeenCalled();
  });

  it('removes the new empty skill when registering its first version fails', async () => {
    jest.spyOn(SkillRegistryApi, 'registerSkill').mockRejectedValue(new Error('Source unreachable'));
    const deleteSkill = jest.spyOn(SkillRegistryApi, 'deleteSkill').mockResolvedValue({});
    renderModal();

    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/main/demo');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(await screen.findByText('Source unreachable')).toBeInTheDocument();
    expect(deleteSkill).toHaveBeenCalledWith('demo', 'acme');
    expect(onRegistered).not.toHaveBeenCalled();
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

  it('explains a folder without SKILL.md', async () => {
    renderModal();
    await chooseUpload();

    await userEvent.upload(screen.getByLabelText('Skill folder'), [folderFile('demo/README.md', '# Demo\n')]);

    expect(await screen.findByText('This folder has no SKILL.md at its top level.')).toBeInTheDocument();
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
});
