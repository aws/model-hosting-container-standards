# Release Runbook

This page outlines the steps to build and publish a new version of this library to PyPI.

Releases for this library are hosted on PyPI here: https://pypi.org/project/model-hosting-container-standards/

## Release Process

1. Prepare release candidate branch
    - Identify the commit hash that you will be releasing. This commit should exist on the main branch.
    - Check the library's version numbers in:
        - `pyproject.toml`
        - `model_hosting_container_standards/__init__.py`
    - If they don't match the version number you are releasing, create a PR to update them.
        - See: [Versioning](#versioning)
    - Now, create a branch whose head is the commit to be released. This could be the main branch.

2. Verify release candidate branch
    - Run the **Build and Publish** workflow against the release candidate branch.
    - Verify that wheel build & test stages pass, then move on to the next step.

3. Tag your release commit
    - After tests pass, create a tag for the commit you are releasing:
    - `git tag -a <vX.X.X> <commit-SHA>`
        - As a reminder, tags are immutable references to specific commits,
          so ensure this commit is ready to be released.
          Tags should not be altered, especially after the version is released to PyPI.
    - The Build & Publish pipeline will run again.
      This time, the wheel will be automatically published to the [test PyPI environment](https://test.pypi.org/project/model-hosting-container-standards/).
    - Verify that everything looks good in the test environment, then proceed.

4. Release the wheel to PyPI
    - After you have verified the release in the test PyPI environment,
      you are ready to release to PyPI.
    - Manually re-run the **Build and Publish** workflow,
      ensuring the **Release wheel to Production PyPI** checkbox is selected.
    - Once again the pipeline will run, this time it will publish to Production PyPI.
    - The package version will be updated on the [PyPI page](https://pypi.org/project/model-hosting-container-standards/) once the release is complete
      and is available in pip once the website is updated.

5. Increment version on main
    - Update the version on the main branch now that we have released the previous version.
      See: [Versioning](#versioning)


## Versioning

This library uses [semantic versioning](https://semver.org).
A quick summary is:

- Library versions are in the form major.minor.patch.
- Update the major version number **X**.X.X for breaking changes.
- Update the minor version number X.**X**.X for backwards compatible features.
- Update the patch version number X.X.**X** for backwards compatible bug-fixes.
