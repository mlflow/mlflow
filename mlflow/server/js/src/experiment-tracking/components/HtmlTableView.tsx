/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { LegacyTable } from '@databricks/design-system';
import './HtmlTableView.css';

type Props = {
  columns: any[];
  values: any[];
  styles?: any;
  testId?: string;
  scroll?: any;
};

export function HtmlTableView({ columns, values, styles = {}, testId, scroll }: Props) {
  return (
    <LegacyTable
      className="mlflow-html-table-view"
      data-testid={testId}
      dataSource={values}
      columns={columns}
      scroll={scroll}
      size="middle"
      pagination={false}
      style={styles}
    />
  );
}
