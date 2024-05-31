package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

type Duration struct {
	time.Duration
}

var ErrDuration = errors.New("invalid duration")

func (d *Duration) UnmarshalJSON(b []byte) error {
	var v interface{}
	if err := json.Unmarshal(b, &v); err != nil {
		return fmt.Errorf("could not unmarshall duration: %w", err)
	}

	switch value := v.(type) {
	case float64:
		d.Duration = time.Duration(value)

		return nil
	case string:
		var err error

		d.Duration, err = time.ParseDuration(value)
		if err != nil {
			return fmt.Errorf("could not parse duration \"%s\": %w", value, err)
		}

		return nil
	default:
		return ErrDuration
	}
}

type Config struct {
	Address             string   `json:"address"`
	DefaultArtifactRoot string   `json:"defaultArtifactRoot"`
	LogLevel            string   `json:"logLevel"`
	PythonAddress       string   `json:"pythonAddress"`
	PythonCommand       []string `json:"pythonCommand"`
	PythonEnv           []string `json:"pythonEnv"`
	ShutdownTimeout     Duration `json:"shutdownTimeout"`
	StaticFolder        string   `json:"staticFolder"`
	StoreURL            string   `json:"storeUrl"`
	Version             string   `json:"version"`
}
