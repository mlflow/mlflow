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

package cmd

import (
	"github.com/spf13/cobra"

	"github.com/operator-framework/operator-sdk/version"
)

func NewRootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "operator-sdk",
		Short:   "An SDK for building operators with ease",
		Version: version.Version,
	}

	cmd.AddCommand(NewNewCmd())
	cmd.AddCommand(NewAddCmd())
	cmd.AddCommand(NewBuildCmd())
	cmd.AddCommand(NewGenerateCmd())
	cmd.AddCommand(NewUpCmd())
	cmd.AddCommand(NewCompletionCmd())
	cmd.AddCommand(NewTestCmd())
	cmd.AddCommand(NewScorecardCmd())
	cmd.AddCommand(NewPrintDepsCmd())
	cmd.AddCommand(NewMigrateCmd())
	cmd.AddCommand(NewRunCmd())
	cmd.AddCommand(NewOLMCatalogCmd())

	return cmd
}
