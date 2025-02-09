import React, { useState } from 'react';
import {
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Typography,
  useDesignSystemTheme,
  MinusSquareIcon,
  PlusSquareIcon,
} from '@databricks/design-system';
import { LogModelWithSignatureUrl } from '../../common/constants';
import { ColumnSpec, TensorSpec, ColumnType } from '../types/model-schema';
import { FormattedMessage } from 'react-intl';
import { Interpolation, Theme } from '@emotion/react';
import { isEmpty } from 'lodash';

const { Text } = Typography;
const INDENTATION_SPACES = 2;

type Props = {
  schema?: any;
  defaultExpandAllRows?: boolean;
};

function getTensorTypeRepr(tensorType: TensorSpec): string {
  return `Tensor (dtype: ${tensorType['tensor-spec'].dtype}, shape: [${tensorType['tensor-spec'].shape}])`;
}

// return a formatted string representation of the column type
function getColumnTypeRepr(columnType: ColumnType, indentationLevel: number): string {
  const { type } = columnType;

  const indentation = ' '.repeat(indentationLevel * INDENTATION_SPACES);
  if (type === 'object') {
    const propertyReprs = Object.keys(columnType.properties).map((propertyName) => {
      const property = columnType.properties[propertyName];
      const requiredRepr = property.required ? '' : ' (optional)';
      const propertyRepr = getColumnTypeRepr(property, indentationLevel + 1);
      const indentOffset = (indentationLevel + 1) * INDENTATION_SPACES;

      return `${' '.repeat(indentOffset)}${propertyName}: ${propertyRepr.slice(indentOffset) + requiredRepr}`;
    });

    return `${indentation}{\n${propertyReprs.join(',\n')}\n${indentation}}`;
  }

  if (type === 'array') {
    const indentOffset = indentationLevel * INDENTATION_SPACES;
    const itemsTypeRepr = getColumnTypeRepr(columnType.items, indentationLevel).slice(indentOffset);
    return `${indentation}Array(${itemsTypeRepr})`;
  }

  return `${indentation}${type}`;
}

function ColumnName({ spec }: { spec: ColumnSpec | TensorSpec }): React.ReactElement {
  let required = true;
  if (spec.required !== undefined) {
    ({ required } = spec);
  } else if (spec.optional !== undefined && spec.optional) {
    required = false;
  }
  const requiredTag = required ? <Text bold>(required)</Text> : <Text color="secondary">(optional)</Text>;

  const name = 'name' in spec ? spec.name : '-';

  return (
    <Text css={{ marginLeft: 32 }}>
      {name} {requiredTag}
    </Text>
  );
}

function ColumnSchema({ spec }: { spec: ColumnSpec | TensorSpec }): React.ReactElement {
  const { theme } = useDesignSystemTheme();
  const repr = spec.type === 'tensor' ? getTensorTypeRepr(spec) : getColumnTypeRepr(spec, 0);

  return (
    <pre
      css={{
        whiteSpace: 'pre-wrap',
        padding: theme.spacing.sm,
        marginTop: theme.spacing.sm,
        marginBottom: theme.spacing.sm,
      }}
    >
      {repr}
    </pre>
  );
}

const SchemaTableRow = ({ schemaData }: { schemaData?: (ColumnSpec | TensorSpec)[] }) => {
  if (isEmpty(schemaData)) {
    return (
      <TableRow>
        <TableCell>
          <FormattedMessage
            defaultMessage="No schema. See <link>MLflow docs</link> for how to include
                     input and output schema with your model."
            description="Text for schema table when no schema exists in the model version
                     page"
            values={{
              link: (chunks: any) => (
                <a href={LogModelWithSignatureUrl} target="_blank" rel="noreferrer">
                  {chunks}
                </a>
              ),
            }}
          />
        </TableCell>
      </TableRow>
    );
  }
  return (
    <>
      {schemaData?.map((schemaRow, index) => (
        <TableRow key={index}>
          <TableCell css={{ flex: 2, alignItems: 'center' }}>
            <ColumnName spec={schemaRow} />
          </TableCell>
          <TableCell css={{ flex: 3, alignItems: 'center' }}>
            <ColumnSchema spec={schemaRow} />
          </TableCell>
        </TableRow>
      ))}
    </>
  );
};

export const SchemaTable = ({ schema, defaultExpandAllRows }: Props) => {
  const { theme } = useDesignSystemTheme();
  const [inputsExpanded, setInputsExpanded] = useState(defaultExpandAllRows);
  const [outputsExpanded, setOutputsExpanded] = useState(defaultExpandAllRows);

  return (
    <Table css={{ maxWidth: 800 }}>
      <TableRow isHeader>
        <TableHeader componentId="mlflow.schema_table.header.name" css={{ flex: 2 }}>
          <Text bold css={{ paddingLeft: theme.spacing.lg + theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Name"
              description="Text for name column in schema table in model version page"
            />
          </Text>
        </TableHeader>
        <TableHeader componentId="mlflow.schema_table.header.type" css={{ flex: 3 }}>
          <Text bold>
            <FormattedMessage
              defaultMessage="Type"
              description="Text for type column in schema table in model version page"
            />
          </Text>
        </TableHeader>
      </TableRow>
      <>
        <TableRow onClick={() => setInputsExpanded(!inputsExpanded)} css={{ cursor: 'pointer' }}>
          <TableCell>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <div
                css={{
                  width: theme.spacing.lg,
                  height: theme.spacing.lg,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  svg: {
                    color: theme.colors.textSecondary,
                  },
                }}
              >
                {inputsExpanded ? <MinusSquareIcon /> : <PlusSquareIcon />}
              </div>
              <FormattedMessage
                defaultMessage="Inputs ({numInputs})"
                description="Input section header for schema table in model version page"
                values={{
                  numInputs: schema.inputs.length,
                }}
              />
            </div>
          </TableCell>
        </TableRow>
        {inputsExpanded && <SchemaTableRow schemaData={schema.inputs} />}
        <TableRow onClick={() => setOutputsExpanded(!outputsExpanded)} css={{ cursor: 'pointer' }}>
          <TableCell>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <div
                css={{
                  width: theme.spacing.lg,
                  height: theme.spacing.lg,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  svg: {
                    color: theme.colors.textSecondary,
                  },
                }}
              >
                {outputsExpanded ? <MinusSquareIcon /> : <PlusSquareIcon />}
              </div>
              <FormattedMessage
                defaultMessage="Outputs ({numOutputs})"
                description="Input section header for schema table in model version page"
                values={{
                  numOutputs: schema.outputs.length,
                }}
              />
            </div>
          </TableCell>
        </TableRow>
        {outputsExpanded && <SchemaTableRow schemaData={schema.outputs} />}
      </>
    </Table>
  );
};
