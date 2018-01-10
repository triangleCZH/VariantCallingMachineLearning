#this is to run the python file to record the shell commands and generate all folder hierarchy
python help_folder_generate.py >> help.sh
chmod 711 help.sh
./help.sh
rm help.sh
