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

  it("should skip a malformed filename without discarding the run's other images", () => {
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
            // Missing the timestamp label -- parseImageFile throws on this.
            path: 'images/broken%step%0%1%UUID.png',
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
    // The valid images survive; only the malformed one is dropped.
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

  it('should record an empty image state for a run whose files are all malformed', () => {
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
    // The run gets an (empty) image state rather than the reducer aborting
    // and leaving the previous state untouched.
    expect(newState).toEqual({ '123': {} });
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
});
