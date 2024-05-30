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
	Address             string   `json:"Address"`
	DefaultArtifactRoot string   `json:"DefaultArtifactRoot"`
	LogLevel            string   `json:"LogLevel"`
	PythonAddress       string   `json:"PythonAddress"`
	PythonCommand       []string `json:"PythonCommand"`
	PythonEnv           []string `json:"PythonEnv"`
	ShutdownTimeout     Duration `json:"ShutdownTimeout"`
	StaticFolder        string   `json:"StaticFolder"`
	StoreURL            string   `json:"StoreURL"`
	Version             string   `json:"Version"`
}
