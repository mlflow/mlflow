package config

import (
	"encoding/json"
	"errors"
	"time"
)

type Duration struct {
	time.Duration
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var v interface{}
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	switch value := v.(type) {
	case float64:
		d.Duration = time.Duration(value)
		return nil
	case string:
		var err error
		d.Duration, err = time.ParseDuration(value)
		if err != nil {
			return err
		}
		return nil
	default:
		return errors.New("invalid duration")
	}
}

type Config struct {
	Address             string
	DefaultArtifactRoot string
	LogLevel            string
	PythonAddress       string
	PythonCommand       []string
	PythonEnv           []string
	ShutdownTimeout     Duration
	StaticFolder        string
	StoreURL            string
	Version             string
}
