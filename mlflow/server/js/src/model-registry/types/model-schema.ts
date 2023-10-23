export type DataType =
  | 'binary'
  | 'datetime'
  | 'boolean'
  | 'double'
  | 'float'
  | 'integer'
  | 'long'
  | 'string';

export type ColumnType = ScalarType | ArrayType | ObjectType;
export type ScalarType = {
  type: DataType;
};
export type ArrayType = {
  type: 'array';
  items: ColumnType;
};
export type ObjectType = {
  type: 'object';
  properties: Map<string, ColumnType>;
};
export type ColumnSpec = ColumnType & {
  name: string;
  optional?: boolean;
};

export type TensorSpec = {
  type: 'tensor';
  'tensor-spec': {
    dtype: string;
    shape: Array<number>;
  };
  optional?: boolean;
};
