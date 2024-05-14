package contract

import "github.com/gofiber/fiber/v2"

type Parser interface {
	ParseBody(ctx *fiber.Ctx, out interface{}) *Error
	ParseQuery(ctx *fiber.Ctx, out interface{}) *Error
}
