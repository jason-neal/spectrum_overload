#!/bin/sh
# pre-commit.sh
# GitHook for running tests before commiting
# To use link it to 
# ln -s pre-commit.sh .git/hooks/pre-commit

git stash -q --keep-index
bash run_tests.sh
RESULT=$?
git stash pop -q
[ $RESULT -ne 0 ] && exit 1
exit 0