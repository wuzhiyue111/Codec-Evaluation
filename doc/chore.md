## Software Packaging and Distribute

Install `build` and generate built distribution.
```
pip install build
python -m build
```

Install `twine` and upload .whl and .tar.gz file.

> To securely upload your project, you'll need a PyPI API token. Create one at https://pypi.org/manage/account/#api-tokens, setting the "Scope" to "Entire account". Don't close the page until you have copied and saved the token — you won't see that token again.

```
pip install twine
twine upload --repository pypi dist/*
```
