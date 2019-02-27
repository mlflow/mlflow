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
	"github.com/operator-framework/operator-sdk/commands/operator-sdk/cmd/generate"

	"github.com/spf13/cobra"
)

func NewGenerateCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "generate <generator>",
		Short: "Invokes specific generator",
		Long:  `The operator-sdk generate command invokes specific generator to generate code as needed.`,
	}
	cmd.AddCommand(generate.NewGenerateK8SCmd())
	cmd.AddCommand(generate.NewGenerateOpenAPICmd())
	return cmd
}
