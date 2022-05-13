import sys
import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    os.system('echo "\033[0;35m"')
    os.system('cd {}'.format(FILE_PATH + "/"))
    os.system('echo \n\n\n\n')
    os.system('echo "> Compiling epscom driver in folder:"')
    os.system('pwd')
    os.system('echo "\033[0;32m"')
    os.system('make')
    os.system('echo "\n> Folder Contents:"')
    os.system('ls')
    os.system('echo "\033[0;35m"')
    os.system('echo "> Compilation Completed"')
    os.system('echo "> Moving Files"')
    os.system('mv epscom-cmd ../')
    os.system('mv epscom-dbg ../')
    os.system('echo "\n> Folder Contents:"')
    os.system('ls')
    os.system('echo "\n> Root Folder Contents:"')
    os.system('ls ../')
    os.system('echo "\033[0m"')
    os.system('echo \n\n\n\n')