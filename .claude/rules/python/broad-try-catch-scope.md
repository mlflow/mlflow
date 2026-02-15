---
paths: "**/*.py"
---

# Broad Try-Catch Scope

Wrap only the specific operations that can raise exceptions. Keep safe operations outside the try block to improve debugging and avoid masking unexpected errors.

```python
# Bad
try:
    never_fails()
    can_fail()
except ...:
    handle_error()

# Good
never_fails()
try:
    can_fail()
except ...:
    handle_error()
```
