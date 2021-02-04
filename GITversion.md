# GIT version
## Commits

Following rules and guidelines are used for committing changes.

When creating commit message, follow these 7 rules (check https://chris.beams.io/posts/git-commit/#seven-rules for further explanation):
    Separate subject from body with a blank line.
    Limit the subject line to 50 characters.
    Capitalize the subject line.
    Do not end the subject line with a period.
    Use the imperative mood in the subject line.
        **Correct:** Fix pre-processing pipeline.
        **Incorrect:** Fixed pre-processing pipeline, Fixing pre-processing pipeline.
    Wrap the body at 72 characters.
    Use the body to explain what and why, not how.

One commit should cover one meaningful change. However, this is relative, so there are some examples:
    **Correct:** If I can describe the change by one sentence (e.g. Reformat code in risk map pipeline , or Add postprocessing for static risk calculation ).
    **Incorrect:** If I need to use conjunctions like Add loading monitoring data and reformat code in postprocessing . In this case, two commits should be done.
    **Reason:** If there is a bug in code or unexpected behavior, we can easily go through the commit messages and filter only "relevant" changes.

Do not commit personal settings or personal notes.
    **Reason:** Sometimes it is hard to distinguish between needed changes and personal settings. 
    **Another reason:** is that your setting is probably not required by other teammates, all of them need to change that according to their preferences.
    Note: If you are using PyCharm for code versioning, you can simply create custom Change list - files moved to this change list are not committed by default.

## Branches

### Following rules and guidelines are used for git branches.

Never commit directly to master  branch. To make changes to master , new branch must be created (from master ) and changes should be applied using pull-requests only.
All branches (except master ) must include prefixes in names:
    prefix feature/  if new feature is going to be added, e.g. feature/sliding-window-integration ,
    prefix fix/  if the only changes that are made are in order to fix something, e.g. fix/api-matching-problem .
Words in branch name are joined by character "-"   instead of character "_" , e.g. fix/api-matching-problem .

Guidelines regarding branching system are commonly used, however you can adjust them according to your project needs (for example to Gitflow strategy).

## Pull-requests

Following rules and guidelines are used for pull-requests.

For pull-request title, follow the same naming conventions as with single commits (section Commits).
Title should be descriptive enough but not too long - to provide more detailed description, use pull-request description instead.
Each pull-request must be reviewed by at least one reviewer (this can vary according to project needs).
Pull-request is merged by last reviewer after all other reviewers approved the changes.
Source branch should be deleted after pull-request is merged.
Use always the same pre-defined merge type for pull-request (e.g. Squash ).

## Code reviews

Why code reviews? Knowledge transfer between members of the team and improvement of the quality of the code.

### Best practices and recommendations

If something is not clear, there is always option to ask author of the code through the comment.
Author of the code should be reminded if the code doesn't comply to the agreed coding conventions.
It is highly encouraged to propose improvements (if reviewer sees any). Examples:
    Repeating code: This code is repeating, consider using separate function. , (Do not Repeat Yourself (DRY) principle).

    I know better/shorter/simpler way how to do the same thing:

        This for loop can be changed to simple list comprehension as follows: 
        variable = [x for x ...]
    Too big functions: This function is big and hard to read, consider split the code into functions. .
    Code that is hard to understand: Why is this computation done here? Consider using comment to describe that in more detail. .
Be kind, always. Probably, behind every change is hidden hard work, appreciate that.
    All comments should be describing the code, not the author: This part of code seems inefficient  instead of You wrote inefficient code here.
    Use recommendations instead commands: Consider using... , or Would not it be better if...? .
Author of the code should realize that the comments are for the purpose of improving the quality of the code, not criticism of the author.
    
## Sources and further reading
- https://phauer.com/2018/code-review-guidelines/#tl-dr
- https://medium.com/palantir/code-review-best-practices-19e02780015f
- https://chris.beams.io/posts/git-commit/
