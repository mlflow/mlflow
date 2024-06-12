package sql

import (
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"time"

	"gorm.io/gorm"

	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store/sql/model"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"
)

func (s Store) GetExperiment(id string) (*protos.Experiment, *contract.Error) {
	idInt, err := strconv.ParseInt(id, 10, 32)
	if err != nil {
		return nil, contract.NewErrorWith(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			fmt.Sprintf("failed to convert experiment id %q to int", id),
			err,
		)
	}

	experiment := model.Experiment{ID: utils.PtrTo(int32(idInt))}
	if err := s.db.Preload("Tags").First(&experiment).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, contract.NewError(
				protos.ErrorCode_RESOURCE_DOES_NOT_EXIST,
				fmt.Sprintf("No Experiment with id=%d exists", idInt),
			)
		}

		return nil, contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			"failed to get experiment",
			err,
		)
	}

	return experiment.ToProto(), nil
}

func (s Store) CreateExperiment(input *protos.CreateExperiment) (string, *contract.Error) {
	experiment := model.NewExperimentFromProto(input)

	if err := s.db.Transaction(func(transaction *gorm.DB) error {
		if err := transaction.Create(&experiment).Error; err != nil {
			return fmt.Errorf("failed to insert experiment: %w", err)
		}

		if utils.IsNilOrEmptyString(experiment.ArtifactLocation) {
			artifactLocation, err := url.JoinPath(s.config.DefaultArtifactRoot, strconv.Itoa(int(*experiment.ID)))
			if err != nil {
				return fmt.Errorf("failed to join artifact location: %w", err)
			}
			experiment.ArtifactLocation = &artifactLocation
			if err := transaction.Model(&experiment).UpdateColumn("artifact_location", artifactLocation).Error; err != nil {
				return fmt.Errorf("failed to update experiment artifact location: %w", err)
			}
		}

		return nil
	}); err != nil {
		if errors.Is(err, gorm.ErrDuplicatedKey) {
			return "", contract.NewError(
				protos.ErrorCode_RESOURCE_ALREADY_EXISTS,
				fmt.Sprintf("Experiment(name=%s) already exists.", *experiment.Name),
			)
		}

		return "", contract.NewErrorWith(protos.ErrorCode_INTERNAL_ERROR, "failed to create experiment", err)
	}

	return strconv.Itoa(int(*experiment.ID)), nil
}

func (s Store) DeleteExperiment(id string) *contract.Error {
	idInt, err := strconv.ParseInt(id, 10, 32)
	if err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			fmt.Sprintf("failed to convert experiment id (%s) to int", id),
			err,
		)
	}

	if err := s.db.Transaction(func(transaction *gorm.DB) error {
		// Update experiment
		uex := transaction.Model(&model.Experiment{}).
			Where("experiment_id = ?", idInt).
			Updates(&model.Experiment{
				LifecycleStage: utils.PtrTo(string(model.LifecycleStageDeleted)),
				LastUpdateTime: utils.PtrTo(time.Now().UnixMilli()),
			})

		if uex.Error != nil {
			return fmt.Errorf("failed to update experiment (%d) during delete: %w", idInt, err)
		}

		if uex.RowsAffected != 1 {
			return gorm.ErrRecordNotFound
		}

		// Update runs
		if err := transaction.Model(&model.Run{}).
			Where("experiment_id = ?", idInt).
			Updates(&model.Run{
				LifecycleStage: utils.PtrTo(string(model.LifecycleStageDeleted)),
				DeletedTime:    utils.PtrTo(time.Now().UnixMilli()),
			}).Error; err != nil {
			return fmt.Errorf("failed to update runs during delete: %w", err)
		}

		return nil
	}); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return contract.NewError(
				protos.ErrorCode_RESOURCE_DOES_NOT_EXIST,
				fmt.Sprintf("No Experiment with id=%d exists", idInt),
			)
		}

		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			"failed to delete experiment",
			err,
		)
	}

	return nil
}
