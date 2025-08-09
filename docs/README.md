# MLflow Documentation

This README covers information about the main MLflow documentation. The API reference is built separately and included as a static folder during the full build process. Please check out the [README](https://github.com/mlflow/mlflow/blob/master/docs/api_reference/README.md) in the `api_reference` folder for more information.

## Prerequisites

**Necessary**

- NodeJS >= 18.0 (see the [NodeJS documentation](https://nodejs.org/en/download) for installation instructions)
- (For building MDX files from `.ipynb` files) Python 3.9+, [nbconvert](https://pypi.org/project/nbconvert/), [nbformat](https://pypi.org/project/nbformat/) and [pyyml](https://pypi.org/project/pyyml/)

**Optional**

- (For building API docs) See [doc-requirements.txt](https://github.com/mlflow/mlflow/blob/master/requirements/doc-requirements.txt) for API doc requirements.

## Installation

```
$ npm install
```

## Local Development

1. If you haven't done this before, run `npm run convert-notebooks` to convert `.ipynb` files to `.mdx` files. The generated files are git-ignored.

2. Run the development server:

```
$ npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

**Note**: Some server-side rendering features will not work in this mode (e.g. the [client redirects plugin](https://docusaurus.io/docs/api/plugins/@docusaurus/plugin-client-redirects)). To test these, please use the "Build and Serve" workflow below.

## Build and Serve

In order to build the full MLflow documentation (i.e. the contents of https://mlflow.org/docs/latest/), please follow the following steps:

1. Run `npm run build-api-docs` in order to build the API reference and copy the generated HTML to `static/api_reference`.
   a. To speed up the build locally, you can run `npm run build-api-docs:no-r` to skip building R documentation
2. Run `npm run convert-notebooks` to convert `.ipynb` files to `.mdx` files. The generated files are git-ignored.
3. **⚠️ Important!** Run `export DOCS_BASE_URL=/docs/latest` (or wherever the docs are expected to be served). This configures the [Docusaurus baseUrl](https://docusaurus.io/docs/api/docusaurus-config#baseUrl), and the site may not render correctly if this is improperly set.
4. Finally, run `npm run build`. This generates static files in the `build` directory, which can then be served.
5. (Optional) To serve the artifacts generated in the above step, run `npm run serve`.

## Building for release

The generated `build` folder is expected to be hosted at https://mlflow.org/docs/latest. However, as our docs are versioned, we also have to generate the documentation for `https://mlflow.org/docs/{version}`. To do this conveniently, you can run the following command:

```
npm run build-all
```

This command will run all the necessary steps from the "Build and Serve" workflow above, and set the correct values for `DOCS_BASE_URL`. The generated HTML will be dumped to `build/latest` and `build/{version}`. These two folders can then be copied to the [docs repo](https://github.com/mlflow/mlflow-legacy-website/tree/main/docs) and uploaded to the website.

## Troubleshooting

### `Error: Invalid sidebar file at "sidebarsGenAI.ts". These sidebar document ids do not exist:`

This error occurs when some links in the sidebar point to non-existent documents.

When it errors for `-ipynb` pages, it is not the link problem but the notebook conversion script is not run. Run `npm run convert-notebooks` in the above steps to convert `.ipynb` files to `.mdx` files. The generated files are git-ignored.

```
[ERROR] Error: Invalid sidebar file at "sidebarsGenAI.ts".
These sidebar document ids do not exist:

eval-monitor/notebooks/huggingface-evaluation-ipynb
eval-monitor/notebooks/question-answering-evaluation-ipynb
...
```


```

This command will run all the necessary steps from the "Build and Serve" workflow above, and set the correct values for `DOCS_BASE_URL`. The generated HTML will be dumped to `build/latest` and `build/{version}`. These two folders can then be copied to the [docs repo](https://github.com/mlflow/mlflow-legacy-website/tree/main/docs) and uploaded to the website.


# Style Guide

## Principles

Following the [Principles for great content](https://www.writethedocs.org/guide/writing/docs-principles/#principles-for-great-content) from Write the Docs, we aim to write documentation that is:

* **Skimmable**: Readers can easily identify and skip over contents that are not relevant to them.
* **Exemplary**: Include examples and hands-on tutorials, not just conceptual explanations.
* **Consistent**: Use consistent terminology, formatting, and structure throughout the documentation.
* **Current**: Keep the documentation up to date. Incorrect information is worse than missing information.

Since MLflow is an open-source project, **documentation is a part of the product** and strongly tied to the user experience.

## Keep it simple

Content length has significant impact on the user engagement. We follow the best practice from [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/top-10-tips-style-voice) to keep the content concise and easy to read.

|Guide|Replace this| With this|
|---|---|---|
|Prefer fewer words.|If you're ready to purchase Office 365 for your organization, contact your Microsoft account representative.|Ready to buy? Contact us.|
|Get to the point fast. Front-load keywords for scanning. |Templates provide a starting point for creating new documents. A template can include the styles, formats, and page layouts you use frequently. Consider creating a template if you often use the same page layout and style for documents.|Save time by creating a document template that includes the styles, formats, and page layouts you use most often. Then use the template whenever you create a new document.|
|Be brief. Prune excess word. |The Recommended Charts command on the Insert tab recommends charts that are likely to represent your data well. Use the command when you want to visually present data, and you're not sure how to do it.|Create a chart that's just right for your data by using the Recommended Charts command on the Insert tab.|
|Revise weak writing. Edit out you can and there is, there are, there were. |You can access Office apps across your devices, and you get online file storage and sharing.|Store files online, access them from all your devices, and share them with coworkers.|

## Carefully consider bullet points

Bullet points are effective tools for presenting lists of items. However, they are also prone to misuse.
* Bullet points implicitly indicates the importance of the items. Having too many bullet points prevent users from finding actually important information.
* Bullet points inserts a mental break for readers. Too many interruption to the reading flow makes it hard to keep the focus.
* Bullet points consumes vertical space and creates unnecessary blank space in the page.


## Recommended Structure

You don't need to follow this structure.

```
- <Product Name>
  |- Quickstart
  |- Concepts
  |- Guides/
  |    |- ... (list of practical how-to guides. e.g., "Searching for Traces")
  |- Integrations/
  |- (Other folders/pages for important topics)
  |- FAQ
```

Then the root `<product-name>/index.mdx` should lays out the high-level overview of the product and navigation to the other pages.

```
# <Product Name>
[1-2 intro sentences about the product.]
[Hero picture of the product]
Short description (less than one scroll) about the use cases of the product.
## Key Benefit
<FeatureHighlights>
   List 4-6 key benefits of the product.
</FeatureHighlights>
## Quickstart
Link to the quickstart page, or the actual quickstart example if it is short enough.
## Guides
Links to important how-to guides or feature specific pages.
... (other contents, but not too many)
```

## Available Custom Components

Reusable components are important for readability. Using custom components in a consistent way helps readers to scan the page and pattern-match the content quickly.

### FeatureHighlights

The `<FeatureHighlights>` component is intended for listing key benefits and features of the product, combined with the `<TileCard>` sub-component: [example](https://mlflow.org/docs/latest/genai/tracing/#what-makes-mlflow-tracing-unique).

```
<FeatureHighlights>
   <TileCard />
   <TileCard />
   ...
</FeatureHighlights>
```

We use the [Lucide icons](https://lucide.dev/icons/) for the icons. Icons are optional, so omit them if not appropriate, e.g., the page already contains many icons in other places.


### Tabs / TabsWrapper

Tabs are great tool for consolidating related contents and reduce the page length, e.g., examples for  same feature but different languages. The builtin `<Tabs>` component creates a border-less tab section. The `<TabsWrapper>` component adds a border around the tab section to make it more distinguishable.

```
<TabsWrapper>
   <Tabs>
      <TabItem value="python" label="Python">
         ...
      </TabItem>
      <TabItem value="typescript" label="TypeScript">
         ...
      </TabItem>
   </Tabs>
</TabsWrapper>
```

**Caveats**: Tabs are great for consolidating contents e.g., examples for same feature but different languages. However, tabs reduce the discoverability of the contents when users are scanning the page. It is not recommended to use tabs for important contents that users should be able to find easily.


### ImageBox

The `<ImageBox>` component add a border and some styling around the image to make it distinct. This component is suitable for two use cases:

* The highlight image on top of the page.
* Putting screenshot of the part of MLflow UI in the middle of texts. Since MLflow UI has white background, sometimes the border between the image and the site background is unclear.


## Caveats of LLM-generated content

LLMs are great tools for generating documentation. However, **do not trust the output blindly**. Even with the best prompt, LLM-generated contents are still prone to errors and biases.

* Most of recent LLM models are biased towards providing verbose information, because of the nature of how they are trained.
* Some models tend to overuse the same patterns, such as bullet points.
* The way human scan the page is different from how LLMs consume texts.
* Unlike code generation, we don't have a good way to automatically ensure the quality of the generated documentation.

Therefore, human review and hand-editing are still necessary.
