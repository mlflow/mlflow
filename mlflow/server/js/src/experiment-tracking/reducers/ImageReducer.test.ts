import { describe, it, expect, jest } from '@jest/globals';
import { imagesByRunUuid } from './ImageReducer';
import type { AsyncFulfilledAction } from '@mlflow/mlflow/src/redux-types';
import type { ListImagesAction } from '@mlflow/mlflow/src/experiment-tracking/actions';

describe('ImageReducer', () => {
  it('should return the initial state', () => {
    const initialState = {};
    const action: AsyncFulfilledAction<ListImagesAction> = {
      type: 'LIST_IMAGES_API_FULFILLED',
      payload: {
        files: [],
        root_uri: '',
      },
    };
    const newState = imagesByRunUuid(initialState, action);
    expect(newState).toEqual({});
  });

  it('should add images to the state', () => {
    const initialState = {};
    const action: AsyncFulfilledAction<ListImagesAction> = {
      type: 'LIST_IMAGES_API_FULFILLED',
      payload: {
        files: [
          {
            path: 'images/image1%step%0%timestamp%1%UUID.png',
            is_dir: false,
            file_size: 123,
          },
          {
            path: 'images/image2%step%1%timestamp%1%UUID.png',
            is_dir: false,
            file_size: 123,
          },
        ],
        root_uri: '',
      },
      meta: {
        id: '123',
        runUuid: '123',
      },
    };
    const newState = imagesByRunUuid(initialState, action);
    expect(newState).toEqual({
      '123': {
        image1: {
          'image1%step%0%timestamp%1%UUID': {
            filepath: 'images/image1%step%0%timestamp%1%UUID.png',
            step: 0,
            timestamp: 1,
          },
        },
        image2: {
          'image2%step%1%timestamp%1%UUID': {
            filepath: 'images/image2%step%1%timestamp%1%UUID.png',
            step: 1,
            timestamp: 1,
          },
        },
      },
    });
  });

  it('should add images to the state with + delimiter', () => {
    const initialState = {};
    const action: AsyncFulfilledAction<ListImagesAction> = {
      type: 'LIST_IMAGES_API_FULFILLED',
      payload: {
        files: [
          {
            path: 'images/image1+step+0+timestamp+1+UUID.png',
            is_dir: false,
            file_size: 123,
          },
          {
            path: 'images/image2+step+1+timestamp+1+UUID.png',
            is_dir: false,
            file_size: 123,
          },
        ],
        root_uri: '',
      },
      meta: {
        id: '123',
        runUuid: '123',
      },
    };
    const newState = imagesByRunUuid(initialState, action);
    expect(newState).toEqual({
      '123': {
        image1: {
          'image1+step+0+timestamp+1+UUID': {
            filepath: 'images/image1+step+0+timestamp+1+UUID.png',
            step: 0,
            timestamp: 1,
          },
        },
        image2: {
          'image2+step+1+timestamp+1+UUID': {
            filepath: 'images/image2+step+1+timestamp+1+UUID.png',
            step: 1,
            timestamp: 1,
          },
        },
      },
    });
  });

  it('should handle error and prevent state update on malformed inputs', () => {
    const initialState = {};
    const action: AsyncFulfilledAction<ListImagesAction> = {
      type: 'LIST_IMAGES_API_FULFILLED',
      payload: {
        files: [
          {
            path: 'images/image1%step%0%1%UUID.png',
            is_dir: false,
            file_size: 123,
          },
        ],
        root_uri: '',
      },
      meta: {
        id: '123',
        runUuid: '123',
      },
    };
    const newState = imagesByRunUuid(initialState, action);
    expect(newState).toEqual({});
  });

  it('should add image and compressed image to the state', () => {
    const initialState = {};
    const action: AsyncFulfilledAction<ListImagesAction> = {
      type: 'LIST_IMAGES_API_FULFILLED',
      payload: {
        files: [
          {
            path: 'images/image1%step%0%timestamp%1%UUID.png',
            is_dir: false,
            file_size: 123,
          },
          {
            path: 'images/image2%step%1%timestamp%1%UUID.png',
            is_dir: false,
            file_size: 123,
          },
          {
            path: 'images/image1%step%0%timestamp%1%UUID.json',
            is_dir: false,
            file_size: 123,
          },
          {
            path: 'images/image2%step%1%timestamp%1%UUID.json',
            is_dir: false,
            file_size: 123,
          },
          {
            path: 'images/image1%step%0%timestamp%1%UUID%compressed.webp',
            is_dir: false,
            file_size: 123,
          },
          {
            path: 'images/image2%step%1%timestamp%1%UUID%compressed.webp',
            is_dir: false,
            file_size: 123,
          },
        ],
        root_uri: '',
      },
      meta: {
        id: '123',
        runUuid: '123',
      },
    };
    const newState = imagesByRunUuid(initialState, action);
    expect(newState).toEqual({
      '123': {
        image1: {
          'image1%step%0%timestamp%1%UUID': {
            filepath: 'images/image1%step%0%timestamp%1%UUID.png',
            compressed_filepath: 'images/image1%step%0%timestamp%1%UUID%compressed.webp',
            step: 0,
            timestamp: 1,
          },
        },
        image2: {
          'image2%step%1%timestamp%1%UUID': {
            filepath: 'images/image2%step%1%timestamp%1%UUID.png',
            compressed_filepath: 'images/image2%step%1%timestamp%1%UUID%compressed.webp',
            step: 1,
            timestamp: 1,
          },
        },
      },
    });
  });

  it('should group video artifacts by key and step alongside their poster', () => {
    const stem = 'rollout+step+3+timestamp+1786553450112+UUID';
    const action: AsyncFulfilledAction<ListImagesAction> = {
      type: 'LIST_IMAGES_API_FULFILLED',
      payload: {
        files: [
          { path: `images/${stem}.mp4`, is_dir: false, file_size: 123 },
          { path: `images/${stem}+compressed.webp`, is_dir: false, file_size: 12 },
        ],
        root_uri: '',
      },
      meta: { id: '123', runUuid: 'run1' },
    } as AsyncFulfilledAction<ListImagesAction>;

    const newState = imagesByRunUuid({}, action);
    const entry = newState['run1']['rollout'][stem];
    expect(entry.filepath).toEqual(`images/${stem}.mp4`);
    expect(entry.compressed_filepath).toEqual(`images/${stem}+compressed.webp`);
    expect(entry.step).toEqual(3);
  });

  it('should keep a video whose poster is absent', () => {
    // The poster is optional, so log_video needs no transcoding dependency.
    const stem = 'rollout+step+0+timestamp+1786553450112+UUID';
    const action: AsyncFulfilledAction<ListImagesAction> = {
      type: 'LIST_IMAGES_API_FULFILLED',
      payload: { files: [{ path: `images/${stem}.mp4`, is_dir: false, file_size: 123 }], root_uri: '' },
      meta: { id: '123', runUuid: 'run1' },
    } as AsyncFulfilledAction<ListImagesAction>;

    const entry = imagesByRunUuid({}, action)['run1']['rollout'][stem];
    expect(entry.filepath).toEqual(`images/${stem}.mp4`);
    expect(entry.compressed_filepath).toBeUndefined();
  });

  it('should ignore artifacts whose extension is neither image nor video', () => {
    const stem = 'rollout+step+0+timestamp+1+UUID';
    const action: AsyncFulfilledAction<ListImagesAction> = {
      type: 'LIST_IMAGES_API_FULFILLED',
      payload: { files: [{ path: `images/${stem}.txt`, is_dir: false, file_size: 1 }], root_uri: '' },
      meta: { id: '123', runUuid: 'run1' },
    } as AsyncFulfilledAction<ListImagesAction>;

    expect(imagesByRunUuid({}, action)['run1']?.['rollout']).toBeUndefined();
  });
});
