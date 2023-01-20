#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Provide pypi token as an argument."
    exit 1
fi

# check that tools are installed
which python
which twine
which git

# check if all changes are commited
git diff --quiet && git diff --cached --quiet
if [ $? -ne 0 ]; then
    echo "There are uncommited changes"
    exit 1
fi

token="$1"

# bump version
sed -i -r 's/version = \"0.0.([.0-9]+)\"/echo "version = \\"0.0.$((\1+1))\\""/ge' pyproject.toml

ver=$(sed -n -r 's/version = \"(0.0.[.0-9]+)\"/\1/p' pyproject.toml)
echo "Update to version $ver"

git add pyproject.toml
git commit -m "Update to version $ver"

# make sure to add git tag:
git tag v$ver master
git push origin master
git push origin v$ver

python -m build
twine upload dist/* --skip-existing -u __token__ -p $token

