#!/bin/zsh
#!/bin/zsh


## Get day number
tail README.md
echo "What number day is it?"
read dayNumber

re='^[0-9]+$'
if ! [[ $dayNumber =~ $re ]];
then
   echo "error: You did not enter an integer."
   exit 1
fi


source activate daysOfCode-env

# Build notebooks for various pedagogical sources.
zsh ipynb_to_md.zsh "pfda"
zsh ipynb_to_md.zsh "cs20"
zsh ipynb_to_md.zsh "homl"


## Write virtual environment to requirements.txt
conda list -e > requirements.txt


conda deactivate



## Add to README
currentDate=$(date +"%B %d, %Y")

MESSAGE_FILE_NAME=".tmp_dialy_message_file.txt"
rm $MESSAGE_FILE_NAME
vim $MESSAGE_FILE_NAME

readmeMessage="
**Day $dayNumber - $currentDate:**
$(cat $MESSAGE_FILE_NAME | sed 's/\. */.\
/g')"
echo $readmeMessage
echo $readmeMessage >> README.md



## Git Commit
commitMessage="$dayNumber of 100 Days of Python
$(cat $MESSAGE_FILE_NAME | fold -w 80 -s)"

rm $MESSAGE_FILE_NAME

echo "\n---\nHere is todays commit message:"
echo $commitMessage
echo "\n"
echo "Shall I commit to git? (y/n)"
read shouldCommit
if [[ $shouldCommit == 'y' ]]
then
	echo "Comitting to git repo..."
	git add --all
	git commit -m $commitMessage
	git push
else
	echo "Nothing commited."
fi

exit


## Get day number
tail README.md
echo "What number day is it?"
read dayNumber

re='^[0-9]+$'
if ! [[ $dayNumber =~ $re ]];
then
   echo "error: You did not enter an integer."
   exit 1
fi


source activate daysOfCode-env

## Build Python for Data Analysis notebooks
echo "Build the Python for Data Analysis notebooks? (y/n)"
read buildPFDA
if [[ $buildPFDA == 'y' ]];
then
	zsh pfda_to_md.zsh
fi

## Build CS 20 notebooks
echo "Build the notebooks for CS 20? (y/n)"
read buildCSTwenty
if [[ $buildCSTwenty == 'y' ]];
then
	zsh cs20_to_md.zsh
fi


## Write virtual environment to requirements.txt
conda list -e > requirements.txt


conda deactivate



## Add to README
currentDate=$(date +"%B %d, %Y")

MESSAGE_FILE_NAME=".tmp_dialy_message_file.txt"
rm $MESSAGE_FILE_NAME
vim $MESSAGE_FILE_NAME

readmeMessage="
**Day $dayNumber - $currentDate:**
$(cat $MESSAGE_FILE_NAME | sed 's/\. */.\
/g')"
echo $readmeMessage
echo $readmeMessage >> README.md



## Git Commit
commitMessage="$dayNumber of 100 Days of Python
$(cat $MESSAGE_FILE_NAME | fold -w 80 -s)"

rm $MESSAGE_FILE_NAME

echo "\n---\nHere is todays commit message:"
echo $commitMessage
echo "\n"
echo "Shall I commit to git? (y/n)"
read shouldCommit
if [[ $shouldCommit == 'y' ]]
then
	echo "Comitting to git repo..."
	git add --all
	git commit -m $commitMessage
	git push
else
	echo "Nothing commited."
fi

exit
