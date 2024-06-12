package sql

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/ncruces/go-sqlite3/gormlite"
	"github.com/sirupsen/logrus"
	"gorm.io/driver/mysql"
	"gorm.io/driver/postgres"
	"gorm.io/driver/sqlserver"
	"gorm.io/gorm"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"

	_ "github.com/ncruces/go-sqlite3/embed" // embed sqlite3 driver
)

type Store struct {
	config *config.Config
	db     *gorm.DB
}

func NewSQLStore(logger *logrus.Logger, config *config.Config) (*Store, error) {
	uri, err := url.Parse(config.StoreURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse store URL %q: %w", config.StoreURL, err)
	}

	var dialector gorm.Dialector

	uri.Scheme, _, _ = strings.Cut(uri.Scheme, "+")

	switch uri.Scheme {
	case "mssql":
		uri.Scheme = "sqlserver"
		dialector = sqlserver.Open(uri.String())
	case "mysql":
		dialector = mysql.Open(fmt.Sprintf("%s@tcp(%s)%s?%s", uri.User, uri.Host, uri.Path, uri.RawQuery))
	case "postgres", "postgresql":
		dialector = postgres.Open(uri.String())
	case "sqlite":
		uri.Scheme = ""
		uri.Path = uri.Path[1:]
		dialector = gormlite.Open(uri.String())
	default:
		return nil, fmt.Errorf("unsupported store URL scheme %q", uri.Scheme) //nolint:err113
	}

	database, err := gorm.Open(dialector, &gorm.Config{
		TranslateError: true,
		Logger:         NewLoggerAdaptor(logger, LoggerAdaptorConfig{}),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database %q: %w", uri.String(), err)
	}

	return &Store{config: config, db: database}, nil
}
