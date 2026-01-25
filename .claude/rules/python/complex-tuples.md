---
paths: "**/*.py"
---

# Complex Tuples

Replace tuples with 3+ elements with named dataclasses. This improves code clarity, prevents positional argument errors, and enables type checking on individual fields.

```python
# Bad
def get_user() -> tuple[str, int, str]:
    return "Alice", 30, "Engineer"


# Good
from dataclasses import dataclass


@dataclass
class User:
    name: str
    age: int
    occupation: str


def get_user() -> User:
    return User(name="Alice", age=30, occupation="Engineer")
```
