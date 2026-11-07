import { afterEach, describe, it, expect, jest } from '@jest/globals';
import { imagesByRunUuid } from './ImageReducer';
import type { AsyncFulfilledAction } from '@mlflow/mlflow/src/redux-types';
import type { ListImagesAction } from '@mlflow/mlflow/src/experiment-tracking/actions';

describe('ImageReducer', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

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

  it('should ignore malformed images when updating the run', () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
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
    expect(newState).toEqual({ '123': {} });
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('image1%step%0%1%UUID.png'));
  });

  it.each(['%', '+'])('should retain valid image pairs around malformed files with %s delimiters', (delimiter) => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const fileKey = ['image1', 'step', '0', 'timestamp', '1', 'UUID'].join(delimiter);
    const filepath = `images/${fileKey}.png`;
    const compressedFilepath = `images/${fileKey}${delimiter}compressed.webp`;
    const initialState = { 'other-run': {} };
    const action: AsyncFulfilledAction<ListImagesAction> = {
      type: 'LIST_IMAGES_API_FULFILLED',
      payload: {
        files: ['images/notes.txt', compressedFilepath, 'images/malformed.png', filepath, 'images'].map((path) => ({
          path,
          is_dir: false,
          file_size: 123,
        })),
        root_uri: '',
      },
      meta: {
        id: '123',
        runUuid: '123',
      },
    };

    expect(imagesByRunUuid(initialState, action)).toEqual({
      'other-run': {},
      '123': {
        image1: {
          [fileKey]: {
            filepath,
            compressed_filepath: compressedFilepath,
            step: 0,
            timestamp: 1,
          },
        },
      },
    });
    expect(initialState).toEqual({ 'other-run': {} });
    expect(warn).toHaveBeenCalledTimes(3);
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
