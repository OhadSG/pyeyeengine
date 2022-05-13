#################################### Dependencies ####################################

pip3_required_packages = {'cython':'any',
                          'numpy': '1.17.4',
                          'parse': 'any',
                          'boto3': 'any'}

apt_required_packages = {'python3-scipy',
                         'python3-sklearn',
                         'python3-skimage',
                         'python3-pandas',
                         'libatlas-base-dev'}

#################################### Dependencies ####################################

import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DEPENDENCIES_FILE_PATH = BASE_PATH + "/../../"
DEPENDENCIES_DOWNLOAD_FOLDER = "packages/"
DEPENDENCIES_DOWNLOAD_FILE_NAME = "requirements_download.sh"
DEPENDENCIES_FILE_NAME = "requirements.txt"
PACKAGE_GRABBER_SCRIPT_PATH = BASE_PATH + "/package_grabber.sh"

def prepare_download_script():
    file_name = DEPENDENCIES_FILE_PATH + DEPENDENCIES_DOWNLOAD_FOLDER + DEPENDENCIES_DOWNLOAD_FILE_NAME
    outFile = open(file_name, "w+")

    outFile.write('echo "Preparing to download packages...."\n')
    outFile.write('echo "-> Copying conf file"\ncp {}/pip.conf /etc/pip.conf\n'.format(BASE_PATH))
    outFile.write('mkdir -p pip3 && cd ./pip3/\n')

    for package in pip3_required_packages:
        packageName = package
        required_version = pip3_required_packages[package]

        if required_version != "any":
            packageName = packageName + "=={}".format(required_version)

        outFile.write('pip3 download {}\n'.format(packageName))

    outFile.write('cd ..\n')
    outFile.write('mkdir -p apt && cd ./apt/\n')
    outFile.write('sudo chmod +x {}\n'.format(PACKAGE_GRABBER_SCRIPT_PATH))

    for package in apt_required_packages:
        outFile.write('sudo {} {}\n'.format(PACKAGE_GRABBER_SCRIPT_PATH, package))

    outFile.close()

def download_packages():
    file_name = DEPENDENCIES_FILE_PATH + DEPENDENCIES_DOWNLOAD_FOLDER + DEPENDENCIES_DOWNLOAD_FILE_NAME
    commands = "cd {}".format(DEPENDENCIES_FILE_PATH + DEPENDENCIES_DOWNLOAD_FOLDER)
    commands += "&& chmod +x {}".format(file_name)
    commands += "&& sudo {}".format(file_name)

    os.system(commands)

def prepare_installation_file():
    with open("{}/../full_version.txt".format(BASE_PATH), 'r') as file:
        version_text = file.read()
        output_file_name = "/engineLog/installation_dump_{}.txt".format(version_text)

    outFile = open(DEPENDENCIES_FILE_PATH + DEPENDENCIES_FILE_NAME, "w+")

    outFile.write('echo "------- Installing pyeyeengine requirements -------" > {}\n'.format(output_file_name))
    outFile.write('echo "-> Configuring DPKG"\n')
    outFile.write('dpkg --configure -a >> {}\n'.format(output_file_name))
    outFile.write('cd ./packages/\n')

    outFile.write('echo "-> Installing apt packages and dependencies"\n')
    outFile.write('sudo dpkg -i ./*.deb >> {}\n'.format(output_file_name))

    outFile.write('sudo apt-get install -fy >> {}\n'.format(output_file_name))

    outFile.write('echo "-> Installing pip packages"\n')
    outFile.write('pip3 install ./*.whl >> {}\n'.format(output_file_name))

    outFile.write('echo "------- Requirements Installation Complete -------" >> {}\n\n'.format(output_file_name))
    outFile.write('echo "-> Installing pyeyeengine"\n')

    outFile.close()

if __name__ == '__main__':
    prepare_download_script()
    download_packages()
    prepare_installation_file()