//nolint:goprintffuncname
package sql

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

type loggerAdaptor struct {
	Logger *logrus.Logger
	Config LoggerAdaptorConfig
}

type LoggerAdaptorConfig struct {
	SlowThreshold             time.Duration
	IgnoreRecordNotFoundError bool
	ParameterizedQueries      bool
}

// NewLoggerAdaptor creates a new logger adaptor.
//
//nolint:ireturn
func NewLoggerAdaptor(l *logrus.Logger, cfg LoggerAdaptorConfig) logger.Interface {
	return &loggerAdaptor{l, cfg}
}

// LogMode implements the gorm.io/gorm/logger.Interface interface and is a no-op.
//
//nolint:ireturn
func (l *loggerAdaptor) LogMode(_ logger.LogLevel) logger.Interface {
	return l
}

const (
	maximumCallerDepth int = 15
	minimumCallerDepth int = 4
)

// getLoggerEntry gets a logger entry with context and caller information added.
func (l *loggerAdaptor) getLoggerEntry(ctx context.Context) *logrus.Entry {
	entry := l.Logger.WithContext(ctx)
	// We want to report the caller of the function that called gorm's logger,
	// not the caller of the loggerAdaptor, so we skip the first few frames and
	// then look for the first frame that is not in the gorm package.
	pcs := make([]uintptr, maximumCallerDepth)
	depth := runtime.Callers(minimumCallerDepth, pcs)
	frames := runtime.CallersFrames(pcs[:depth])

	for f, again := frames.Next(); again; f, again = frames.Next() {
		if !strings.HasPrefix(f.Function, "gorm.io/gorm") {
			entry = entry.WithFields(logrus.Fields{
				"app_file": fmt.Sprintf("%s:%d", f.File, f.Line),
				"app_func": f.Function + "()",
			})

			break
		}
	}

	return entry
}

// Info logs message at info level and implements the gorm.io/gorm/logger.Interface interface.
func (l *loggerAdaptor) Info(ctx context.Context, format string, args ...interface{}) {
	l.getLoggerEntry(ctx).Infof(format, args...)
}

// Warn logs message at warn level and implements the gorm.io/gorm/logger.Interface interface.
func (l *loggerAdaptor) Warn(ctx context.Context, format string, args ...interface{}) {
	l.getLoggerEntry(ctx).Warnf(format, args...)
}

// Error logs message at error level and implements the gorm.io/gorm/logger.Interface interface.
func (l *loggerAdaptor) Error(ctx context.Context, format string, args ...interface{}) {
	l.getLoggerEntry(ctx).Errorf(format, args...)
}

const NanosecondsPerMillisecond = 1e6

// getLoggerEntryWithSQL gets a logger entry with context, caller information and SQL information added.
func (l *loggerAdaptor) getLoggerEntryWithSQL(
	ctx context.Context,
	elapsed time.Duration,
	fc func() (sql string, rowsAffected int64),
) *logrus.Entry {
	entry := l.getLoggerEntry(ctx)

	if fc != nil {
		sql, rows := fc()
		entry = entry.WithFields(logrus.Fields{
			"elapsed": fmt.Sprintf("%.3fms", float64(elapsed.Nanoseconds())/NanosecondsPerMillisecond),
			"rows":    rows,
			"sql":     sql,
		})

		if rows == -1 {
			entry = entry.WithField("rows", "-")
		}
	}

	return entry
}

// Trace logs SQL statement, amount of affected rows, and elapsed time.
// It implements the gorm.io/gorm/logger.Interface interface.
func (l *loggerAdaptor) Trace(
	ctx context.Context,
	begin time.Time,
	function func() (sql string, rowsAffected int64),
	err error,
) {
	if l.Logger.GetLevel() <= logrus.FatalLevel {
		return
	}

	// This logic is similar to the default logger in gorm.io/gorm/logger.
	elapsed := time.Since(begin)

	switch {
	case err != nil &&
		l.Logger.IsLevelEnabled(logrus.ErrorLevel) &&
		(!errors.Is(err, gorm.ErrRecordNotFound) || !l.Config.IgnoreRecordNotFoundError):
		l.getLoggerEntryWithSQL(ctx, elapsed, function).WithError(err).Error("SQL error")
	case elapsed > l.Config.SlowThreshold &&
		l.Config.SlowThreshold != 0 &&
		l.Logger.IsLevelEnabled(logrus.WarnLevel):
		l.getLoggerEntryWithSQL(ctx, elapsed, function).Warnf("SLOW SQL >= %v", l.Config.SlowThreshold)
	case l.Logger.IsLevelEnabled(logrus.DebugLevel):
		l.getLoggerEntryWithSQL(ctx, elapsed, function).Debug("SQL trace")
	}
}
