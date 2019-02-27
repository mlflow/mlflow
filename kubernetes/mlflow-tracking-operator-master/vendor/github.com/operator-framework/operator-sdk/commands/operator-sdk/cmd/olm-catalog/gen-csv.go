// Copyright 2018 The Operator-SDK Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package catalog

import (
	"fmt"
	"path/filepath"

	"github.com/operator-framework/operator-sdk/internal/util/projutil"
	"github.com/operator-framework/operator-sdk/pkg/scaffold"
	"github.com/operator-framework/operator-sdk/pkg/scaffold/input"
	catalog "github.com/operator-framework/operator-sdk/pkg/scaffold/olm-catalog"

	"github.com/coreos/go-semver/semver"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var (
	csvVersion    string
	csvConfigPath string
)

func NewGenCSVCmd() *cobra.Command {
	genCSVCmd := &cobra.Command{
		Use:   "gen-csv",
		Short: "Generates a Cluster Service Version yaml file for the operator",
		Long: `The gen-csv command generates a Cluster Service Version (CSV) YAML manifest
for the operator. This file is used to publish the operator to the OLM Catalog.

A CSV semantic version is supplied via the --csv-version flag.

Configure CSV generation by writing a config file 'deploy/olm-catalog/csv-config.yaml`,
		RunE: genCSVFunc,
	}

	genCSVCmd.Flags().StringVar(&csvVersion, "csv-version", "", "Semantic version of the CSV")
	genCSVCmd.MarkFlagRequired("csv-version")
	genCSVCmd.Flags().StringVar(&csvConfigPath, "csv-config", "", "Path to CSV config file. Defaults to deploy/olm-catalog/csv-config.yaml")

	return genCSVCmd
}

func genCSVFunc(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return fmt.Errorf("command %s doesn't accept any arguments", cmd.CommandPath())
	}

	if err := verifyGenCSVFlags(); err != nil {
		return err
	}

	absProjectPath := projutil.MustGetwd()
	cfg := &input.Config{
		AbsProjectPath: absProjectPath,
		ProjectName:    filepath.Base(absProjectPath),
	}
	if projutil.IsOperatorGo() {
		cfg.Repo = projutil.CheckAndGetProjectGoPkg()
	}

	log.Infof("Generating CSV manifest version %s", csvVersion)

	s := &scaffold.Scaffold{}
	err := s.Execute(cfg,
		&catalog.CSV{CSVVersion: csvVersion, ConfigFilePath: csvConfigPath},
		&catalog.ConcatCRD{ConfigFilePath: csvConfigPath},
	)
	if err != nil {
		return fmt.Errorf("catalog scaffold failed: (%v)", err)
	}
	return nil
}

func verifyGenCSVFlags() error {
	v, err := semver.NewVersion(csvVersion)
	if err != nil {
		return fmt.Errorf("%s is not a valid semantic version: (%v)", csvVersion, err)
	}
	// Ensures numerical values composing csvVersion don't contain leading 0's,
	// ex. 01.01.01
	if v.String() != csvVersion {
		return fmt.Errorf("provided CSV version %s contains bad values (parses to %s)", csvVersion, v)
	}
	return nil
}
