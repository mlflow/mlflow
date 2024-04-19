package main

import (
	"bufio"
	"bytes"
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

func dumpAST(fset *token.FileSet, node *ast.File) {
	// A buffer to store the output
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, node); err != nil {
		panic(err)
	}

	// Output the generated code
	println(buf.String())
}

func mkImportSpec(value string) *ast.ImportSpec {
	return &ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: value}}
}

var importStatements = &ast.GenDecl{
	Tok: token.IMPORT,
	Specs: []ast.Spec{
		mkImportSpec(`"github.com/gofiber/fiber/v2"`),
		mkImportSpec(`"github.com/mlflow/mlflow/mlflow/go/pkg/protos"`),
		mkImportSpec(`"github.com/mlflow/mlflow/mlflow/go/pkg/protos/artifacts"`),
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

func mkNamedField(name string, typ ast.Expr) *ast.Field {
	return &ast.Field{
		Names: []*ast.Ident{ast.NewIdent(name)},
		Type:  typ,
	}
}

func mkField(typ ast.Expr) *ast.Field {
	return &ast.Field{
		Type: typ,
	}
}

func mkMethodInfoInputPointerType(methodInfo server.MethodInfo) *ast.StarExpr {
	return mkStarExpr(mkSelectorExpr(methodInfo.PackageName, methodInfo.Input))
}

// Generate a method declaration on an service interface
func mkServiceInterfaceMethod(methodInfo server.MethodInfo) *ast.Field {
	return &ast.Field{
		Names: []*ast.Ident{ast.NewIdent(strcase.ToCamel(methodInfo.Name))},
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					mkNamedField("input", mkMethodInfoInputPointerType(methodInfo)),
				},
			},
			Results: &ast.FieldList{
				List: []*ast.Field{
					mkField(mkSelectorExpr(methodInfo.PackageName, methodInfo.Output)),
					mkField(mkStarExpr(ast.NewIdent("MlflowError"))),
				},
			},
		},
	}
}

// Generate a service interface declaration
func mkServiceInterfaceNode(serviceInfo server.ServiceInfo) *ast.GenDecl {
	methods := make([]*ast.Field, len(serviceInfo.Methods))
	for idx := range len(serviceInfo.Methods) {
		methods[idx] = mkServiceInterfaceMethod(serviceInfo.Methods[idx])
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

func mkCallExpr(fun ast.Expr, args ...ast.Expr) *ast.CallExpr {
	return &ast.CallExpr{
		Fun:  fun,
		Args: args,
	}
}

// Shorthand for creating &expr
func mkAmpExpr(expr ast.Expr) *ast.UnaryExpr {
	return &ast.UnaryExpr{
		Op: token.AND,
		X:  expr,
	}
}

// err != nil
var errNotEqualNil = &ast.BinaryExpr{
	X:  ast.NewIdent("err"),
	Op: token.NEQ,
	Y:  ast.NewIdent("nil"),
}

// return err
var returnErr = &ast.ReturnStmt{
	Results: []ast.Expr{ast.NewIdent("err")},
}

func mkBlockStmt(stmts ...ast.Stmt) *ast.BlockStmt {
	return &ast.BlockStmt{
		List: stmts,
	}
}

func mkIfStmt(init ast.Stmt, cond ast.Expr, body *ast.BlockStmt) *ast.IfStmt {
	return &ast.IfStmt{
		Init: init,
		Cond: cond,
		Body: body,
	}
}

func mkAssignStmt(lhs []ast.Expr, rhs []ast.Expr) *ast.AssignStmt {
	return &ast.AssignStmt{
		Lhs: lhs,
		Tok: token.DEFINE,
		Rhs: rhs,
	}
}

func mkAppRoute(method server.MethodInfo, endpoint server.Endpoint) ast.Stmt {
	urlExpr := &ast.BasicLit{Kind: token.STRING, Value: fmt.Sprintf(`"/api/v2.0/mlflow/%s/%s"`, method.Name, endpoint.GetFiberPath())}

	// var input *protos.SearchExperiments
	inputExpr := &ast.DeclStmt{
		Decl: &ast.GenDecl{
			Tok: token.VAR,
			Specs: []ast.Spec{
				&ast.ValueSpec{
					Names: []*ast.Ident{ast.NewIdent("input")},
					Type:  mkMethodInfoInputPointerType(method),
				},
			},
		},
	}

	// if err := ctx.QueryParser(&input); err != nil {
	var extractModel ast.Expr
	if endpoint.Method == "GET" {
		extractModel = mkCallExpr(mkSelectorExpr("ctx", "QueryParser"), mkAmpExpr(ast.NewIdent("input")))
	} else {
		extractModel = mkCallExpr(mkSelectorExpr("ctx", "BodyParser"), mkAmpExpr(ast.NewIdent("input")))
	}

	inputErrorCheck := mkIfStmt(mkAssignStmt([]ast.Expr{ast.NewIdent("err")}, []ast.Expr{extractModel}), errNotEqualNil, mkBlockStmt(returnErr))

	// output, err := service.SearchExperiments(input)
	outputExpr := mkAssignStmt([]ast.Expr{
		ast.NewIdent("output"),
		ast.NewIdent("err"),
	}, []ast.Expr{
		mkCallExpr(mkSelectorExpr("service", strcase.ToCamel(method.Name)), ast.NewIdent("input")),
	})

	// if err != nil {
	outputErrorCheck := mkIfStmt(nil, errNotEqualNil, mkBlockStmt(returnErr))

	// return ctx.JSON(&output)
	returnExpr := &ast.ReturnStmt{
		Results: []ast.Expr{
			mkCallExpr(mkSelectorExpr("ctx", "JSON"), mkAmpExpr(ast.NewIdent("output"))),
		},
	}

	// func(ctx *fiber.Ctx) error { .. }
	funcExpr := &ast.FuncLit{
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					mkNamedField("ctx", mkStarExpr(mkSelectorExpr("fiber", "Ctx"))),
				},
			},
			Results: &ast.FieldList{
				List: []*ast.Field{
					mkField(ast.NewIdent("error")),
				},
			},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				inputExpr,
				inputErrorCheck,
				outputExpr,
				outputErrorCheck,
				returnExpr,
			},
		},
	}

	return &ast.ExprStmt{
		// app.Get("/api/v2.0/mlflow/experiments/search", func(ctx *fiber.Ctx) error { .. })
		X: mkCallExpr(
			mkSelectorExpr("app", strcase.ToCamel(endpoint.Method)), urlExpr, funcExpr,
		),
	}
}

func mkRouteRegistrationFunction(serviceInfo server.ServiceInfo) *ast.FuncDecl {
	routes := make([]ast.Stmt, 0, len(serviceInfo.Methods))

	for _, method := range serviceInfo.Methods {
		for _, endpoint := range method.Endpoints {
			routes = append(routes, mkAppRoute(method, endpoint))
		}
	}

	return &ast.FuncDecl{
		Name: ast.NewIdent(fmt.Sprintf("register%sRoutes", serviceInfo.Name)),
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					mkNamedField("service", ast.NewIdent(serviceInfo.Name)),
					mkNamedField("app", mkStarExpr(ast.NewIdent("fiber.App"))),
				},
			},
		},
		Body: &ast.BlockStmt{
			List: routes,
		},
	}
}

func generateServices() {
	decls := []ast.Decl{importStatements}

	services := server.GetServiceInfos()
	for _, serviceInfo := range services {
		decls = append(decls, mkServiceInterfaceNode(serviceInfo))
	}

	for _, serviceInfo := range services {
		decls = append(decls, mkRouteRegistrationFunction(serviceInfo))
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
	// dumpAST(fset, file)
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

			if strings.Contains(tagValue, "query:") {
				continue
			}

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
	generateServices()
}
