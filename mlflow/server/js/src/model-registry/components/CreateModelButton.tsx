import React from 'react';
import { Button } from '@databricks/design-system';
import { CreateModelModal } from './CreateModelModal';
import { FormattedMessage } from 'react-intl';

type Props = {
  buttonType?: string;
  buttonText?: React.ReactNode;
};

type State = any;

export class CreateModelButton extends React.Component<Props, State> {
  state = {
    modalVisible: false,
  };

  hideModal = () => {
    this.setState({ modalVisible: false });
  };

  showModal = () => {
    this.setState({ modalVisible: true });
  };

  render() {
    const { modalVisible } = this.state;
    const buttonType = this.props.buttonType || 'primary';
    const buttonText = this.props.buttonText || (
      <FormattedMessage
        defaultMessage='Create Model'
        description='Create button to register a new model'
      />
    );

    return (
      <div css={styles.wrapper}>
        <Button
          className={`create-model-btn`}
          css={styles.getButtonSize(buttonType)}
          // @ts-expect-error TS(2322): Type 'string' is not assignable to type '"link" | ... Remove this comment to see the full error message
          type={buttonType}
          onClick={this.showModal}
        >
          {buttonText}
        </Button>
        <CreateModelModal modalVisible={modalVisible} hideModal={this.hideModal} />
      </div>
    );
  }
}

const styles = {
  getButtonSize: (buttonType: any) =>
    buttonType === 'primary'
      ? {
          height: '40px',
          width: 'fit-content',
        }
      : { padding: '0px' },
  wrapper: { display: 'inline' },
};
