package main

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/iancoleman/strcase"
	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
)

var importStatements = &ast.GenDecl{
	Tok: token.IMPORT,
	Specs: []ast.Spec{
		// &ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: `"encoding/json"`}},
		// &ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: `"fmt"`}},
		// &ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: `"net/http"`}},
		// &ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: `"net/http/httputil"`}},
		// &ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: `"net/url"`}},
		&ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: `"github.com/mlflow/mlflow/mlflow/go/pkg/protos"`}},
		&ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: `"github.com/mlflow/mlflow/mlflow/go/pkg/protos/artifacts"`}},
	},
}

func mkStarExpr(e ast.Expr) *ast.StarExpr {
	return &ast.StarExpr{
		X: e,
	}
}

func mkSelectorExpr(x string, sel string) *ast.SelectorExpr {
	return &ast.SelectorExpr{X: ast.NewIdent(x), Sel: ast.NewIdent(sel)}
}

func mkMethod(methodInfo server.MethodInfo) *ast.Field {
	return &ast.Field{
		Names: []*ast.Ident{ast.NewIdent(strcase.ToCamel(methodInfo.Name))},
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					{
						Names: []*ast.Ident{ast.NewIdent("input")},
						Type:  mkStarExpr(mkSelectorExpr(methodInfo.PackageName, methodInfo.Input)),
					},
				},
			},
			Results: &ast.FieldList{
				List: []*ast.Field{
					{
						Type: &ast.SelectorExpr{X: ast.NewIdent(methodInfo.PackageName), Sel: ast.NewIdent(methodInfo.Output)},
					},
					{
						Type: mkStarExpr(ast.NewIdent("MlflowError")),
					},
				},
			},
		},
	}
}

func mkServiceNode(serviceInfo server.ServiceInfo) *ast.GenDecl {
	methods := make([]*ast.Field, len(serviceInfo.Methods))
	for idx := range len(serviceInfo.Methods) {
		methods[idx] = mkMethod(serviceInfo.Methods[idx])
	}

	// Create an interface declaration
	return &ast.GenDecl{
		Tok: token.TYPE, // Specifies a type declaration
		Specs: []ast.Spec{
			&ast.TypeSpec{
				Name: ast.NewIdent(serviceInfo.Name), // Interface name
				Type: &ast.InterfaceType{
					Methods: &ast.FieldList{
						List: methods,
					},
				},
			},
		},
	}
}

func saveASTToFile(fset *token.FileSet, file *ast.File, outputPath string) {
	// Create or truncate the output file
	outputFile, err := os.Create(outputPath)
	if err != nil {
		panic(err)
	}

	// Use a bufio.Writer for buffered writing (optional)
	writer := bufio.NewWriter(outputFile)

	// Write the generated code to the file
	if err := format.Node(writer, fset, file); err != nil {
		panic(err)
	}

	// Flush the writer and close the file
	writer.Flush()
	outputFile.Close()

	// Format the generated code file
	cmd := exec.Command("go", "fmt", outputPath)

	// Execute the command
	if output, err := cmd.CombinedOutput(); err != nil {
		log.Fatalf("Failed to format the file: %s, error: %s", output, err)
	}
}

func generateServices() {
	decls := []ast.Decl{importStatements}

	services := server.GetServiceInfos()
	for _, serviceInfo := range services {
		decls = append(decls, mkServiceNode(serviceInfo))
	}

	// Set up the FileSet and the AST File
	fset := token.NewFileSet()

	file := &ast.File{
		Name:  ast.NewIdent("server"),
		Decls: decls,
	}

	// Get the current working directory
	workingDir, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	outputPath := filepath.Join(workingDir, "..", "..", "pkg", "server", "interface.g.go")

	saveASTToFile(fset, file, outputPath)
}

var jsonFieldTagRegexp = regexp.MustCompile(`json:"([^"]+)"`)

func addQueryAnnotation(generatedGoFile string) {
	// Parse the file into an AST
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, generatedGoFile, nil, parser.ParseComments)
	if err != nil {
		panic(err)
	}

	// Create an AST inspector to modify specific struct tags
	ast.Inspect(node, func(n ast.Node) bool {
		// Look for struct type declarations
		ts, ok := n.(*ast.TypeSpec)
		if !ok {
			return true
		}
		st, ok := ts.Type.(*ast.StructType)
		if !ok {
			return true
		}

		// Iterate over fields in the struct
		for _, field := range st.Fields.List {
			if field.Tag == nil {
				continue
			}
			tagValue := field.Tag.Value
			fmt.Println(tagValue)
			matches := jsonFieldTagRegexp.FindStringSubmatch(tagValue)
			if len(matches) > 0 {
				// Modify the tag here
				// The json annotation could be something like `json:"key,omitempty"`
				// We only want the key part, the `omitempty` is not relevant for the query annotation
				key := matches[1]
				if strings.Contains(key, ",") {
					key = strings.Split(key, ",")[0]
				}
				newTag := fmt.Sprintf("`%s query:\"%s\"`", tagValue[1:len(tagValue)-1], key)
				fmt.Println(newTag)
				field.Tag.Value = newTag
			}
		}
		return false
	})

	saveASTToFile(fset, node, generatedGoFile)
}

func addQueryAnnotations() {
	// Get the current working directory
	workingDir, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	protoFolder := filepath.Join(workingDir, "..", "..", "pkg", "protos")
	err = filepath.WalkDir(protoFolder, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if filepath.Ext(path) == ".go" {
			addQueryAnnotation(path)
		}
		return nil
	})

	if err != nil {
		panic(err)
	}
}

func main() {
	addQueryAnnotations()
}
