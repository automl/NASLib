# Contributing to NASLib

- [Tools](#tools)
  - [Pre-Commit Hooks](#pre-commit-hooks)
  - [Gitlint](#gitlint)
- [Styleguide](#styleguide)
  - [Commit Message Guidelines](#commit-message-format)

## Tools

The listed tools here are meant to be supportive for you as a developer and help as guidance to submit good quality code. However, they are optional and not developed by ourselves.

### Pre-Commit Hooks

Git hook scripts are useful for identifying simple issues before submission to code review. Pre-commit hooks run before committing and are used for fixing minor problems, linting or type checking. See [.pre-commit-config.yaml](/.pre-commit-config.yaml) for our configuration and the [pre-commit website][pre-commit] for further information.

Installation:

```bash
pip install pre-commit==3.0.3
pre-commit install
```

### Gitlint

A commit message linter, which enforces some rules of the defined [commit message format](#commit-message-format). You can find further information on [their website][gitlint] as well as [our configuration](/.gitlint).
It can be used as `commit-msg hook`, which prevents you from committing using the wrong format:

```bash
pip install gitlint==0.18.0
gitlint install-hook
# gitlint uninstall-hook
```

## Styleguide

### Commit Message Format

*This specification is inspired by the commit message format of [AngularJS][angular-commits] as well as [conventional commits][conventional-commits].*

#### TL:DR

- Use the present tense (*"add feature"* not *"added feature"*)
- Use the imperative mood (*"move cursor to..."* not *"moves cursor to..."*)
- Commit header consists of a [`type`](#types)) and an uncapitalized [`summary`](#subject)
- Limit the header to 72 characters
- Limit any line of the commit message to 100 characters
- Motivate the change in the `body`
- Reference issues and pull requests in your footer


#### More in-depth

Each commit message consists of a **header**, an optional  [**body**](#body), and an optional [**footer**](#footer), separated by a blank line each.
The message header consists of a [**type**](#types), an optional [**scope**](#scopes) and a [**subject**](#subject).

The header is limited to 72 characters, while any other line cannot exceed 100 characters. This allows for better readability in GitHub as well as respective tools.

```
<type>(<optional scope>): <subject>
<BLANK LINE>
<optional body>
<BLANK LINE>
<optional footer(s)>
```

##### Types

| Type      | Description |
| ---       | --- |
|`feat`     | Commits, which add new features |
|`fix`      | Commits, which fix a bug |
|`docs`     | Commits, that only affect documentation |
|`refactor` | Commits, which change the stucture of the code, however do not change its behaviour |
|`perf`     | Special `refactor` commits, which improve performance |
|`style`     | Commits, that do not affect the meaning (white-space, formatting, missing semi-colons, etc) |
|`build`     | Commits, that affect build components like build tool, ci pipeline, dependencies, project version, ... |
|`test`     | Commits, that add missing tests or correcting existing tests |
|`chore`     | Miscellaneous commits e.g. modifying `.gitignore` |
|`revert`| see chapter [Revert](#revert) |

##### Scopes

Scopes are project specific and will be defined later.

##### Subject

The `subject` contains a succinct description of the change.

- Use the imperative, present tense: "change" not "changed" nor "changes".
  - Think of `This commit will <subject>`
- Don't capitalize the first letter
- No dot (.) at the end

##### Body

The `body` should explain the motivation behind the change. It can include a comparison between new and previous behaviour.
It can consist of multiple newline separated paragraphs. As in `subject`, use the imperative, present tense.
It is an optional part.

##### Footer

The `footer` contains information of breaking changes and deprecation, references to e.g. GitHub issues or PRs as well as other metadata (e.g. [trailers](https://git.wiki.kernel.org/index.php/CommitMessageConventions)).
To separate footer from the body, especially for parsing, the [specifications 8. - 10. of conventional commit](https://www.conventionalcommits.org/en/v1.0.0/#specification) exist<sup>[1](#footnote1)</sup>.

```
{BREAKING CHANGE, DEPRICATED}: <summary>
<description + migration instructions or update path>
<BLANK LINE>
Close #<id>
Reviewed-by: xyz
```

##### Revert

If the commit reverts a previous commit, its header should begin with `revert: `, followed by the header of the reverted commit.
In the body it should say: `This reverts commit <hash>.`, where the hash is the SHA of the commit being reverted.

___

<a name="footnote1">1</a>: Important note: `BREAKING CHANGE` is the only exception for specification 9., where a `<space>` between words is not replaced by a `-`

[angular-commits]: https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit#
[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/
[pre-commit]: https://pre-commit.com/
[gitlint]: https://jorisroovers.com/gitlint/
