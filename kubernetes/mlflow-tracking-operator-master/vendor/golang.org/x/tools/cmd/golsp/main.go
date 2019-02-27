// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// -----------------------------------------------------------------
// WARNING: golsp has been renamed to gopls (see cmd/gopls/main.go).
// This file will be deleted soon.
// -----------------------------------------------------------------

// The golsp command is an LSP server for Go.
// The Language Server Protocol allows any text editor
// to be extended with IDE-like features;
// see https://langserver.org/ for details.
package main // import "golang.org/x/tools/cmd/golsp"

import (
	"context"
	"os"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/tool"
)

func main() {
	tool.Main(context.Background(), &cmd.Application{}, os.Args[1:])
}
