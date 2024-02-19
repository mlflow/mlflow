# How to lint python files using `clint` on VSCode

1. Install [the Pylint extension](https://marketplace.visualstudio.com/items?itemName=ms-python.pylint)
2. Add the following settings to your `settings.json` file (you can open it by pressing `Ctrl + ,`)

```json
{
  "pylint.path": ["${interpreter}", "-m", "clint"]
}
```
