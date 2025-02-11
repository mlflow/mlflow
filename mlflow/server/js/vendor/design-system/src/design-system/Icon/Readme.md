# Icons

## Adding new Icons

Place your svg files into the `/assets/icons` folder.
Name them in UpperCamelCase + Icon, example: `MyNameIcon.svg`.

### Generate .tsx file, interfaces and add to story

```
cd ~/universe/design-system
yarn generate
```

This will:

- Generate tsx components from the svg files in `assets/icons` using @svgr/cli and the template in `generator/iconTemplate.js`
- Generate a list of icons that we can use in stories. See `./generated/stories/IconsList.tsx`
- Export the icons from `generated/icons/index.ts`
- Clean up all generated code using prettier.

### Importing from Figma

Raw icons come in snake-case and do not have fill set to "currentColor".

For this I execute a small Python script:

```python
for count, filename in enumerate(os.listdir(folder)):
    dstExt = filename.split(".")[1]
    parts = [x.title() for x in filename.split(".")[0].split("-")]
    pascalName = ''.join(x for x in parts if not x.isspace())
    dst =folder_out + "/"  + pascalName + "Icon." + dstExt

    src =folder + "/" + filename
    print(src + "->" + dst)

    copyfile(src, dst)
```

and replace `#64727D` with `currentColor`.
