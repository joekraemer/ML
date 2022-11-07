## THINGS TO ALWAYS REMEMBER

dont ever repeat yourself.
figure it out the correct way.
use type hints all over the place
hydra to organize configuration files
use meadowrun to use aws resources
log final dataframes (and perhaps other object) before sending them to be graphed so that graphing can be reworked

## Assignment 2 Reflection

collect thoughts on what went poorly and figure out ways to improve them for later in the class ( and potentially other
classes ).
Better structures that work with distributed systems (ie AWS).

- meadowrun

Better architecture.

- learn how to use dependency injection (?)
- I want to be able to run one specific test.

maybe you should just build core classes for other assignments.
learn how to log.

- specifically key configurations parameters

Get better with pycharm and the tools it has such as the console.
Learn how to launch files with debugging parameters.

## Things to do

1. Figure out if I can get dataframes back from meadowrun/AWS

## Meadowrun Notes

- meadowrun can't use conda, must use pip or poetry

It sounds like you're thinking about how to operate on your local data from the cloud?
One option is to include it as part of your "code" if it's not too large using the
working_directory_globs parameter on mirror_local. The other main option is to just
upload it to S3 using the AWS CLI and then fetch it in your remote code using boto3.
The only wrinkle there is needing to grant permissions to access whatever S3
bucket you use: https://docs.meadowrun.io/en/stable/how_to/access_resources/