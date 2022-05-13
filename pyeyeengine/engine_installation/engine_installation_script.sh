#!/usr/bin/env bash

printf "\n\n\n\n---- START OF BUILD SCRIPT ----\n\n\n"

# Create WHL on remote

REQUIRED_BRANCH="qa"
CONFIGURATION="-qa"
WHL_FILE_COMPILER="192.168.0.70"
VERSION=$(cat $WORKSPACE/pyeyeengine/version.txt)

echo
echo
echo "> Performing operations on host"

ssh -T root@$WHL_FILE_COMPILER << EOSSH || { printf "\n\n*** SSH Connection Failed. Code: $? ***\n\n\n" ; exit 99; }
  echo "> Connected to host"
  rm EnginePackage*
  rm -rf ~/engine_dev
  rm -rf ~/engine_package
  mkdir -p ~/engine_dev
  mkdir -p ~/engine_package
  cd ~/engine_dev
  echo "> Cloning from $REQUIRED_BRANCH"
  git clone -b $REQUIRED_BRANCH https://SergeyEyeclick:0Virus95@bitbucket.org/eyeclickdev/pyeyeengine.git || { printf "\n\n*** Cloning Failed. Code: $? ***\n\n\n"; exit 1; }
  cd ~/engine_dev/pyeyeengine
  echo "> Killing pyeyeengine-server & python"
  killall pyeyeengine-server
  pkill python
  echo "> Compiling WHL File"
  env BUILD_NUMBER="$BUILD_NUMBER$CONFIGURATION" python3 setup_wheel.py bdist_wheel
  cd dist;
  cp *.whl ~/engine_package
  echo "> Creating EnginePackage and install scripts"
  python3 ~/engine_dev/pyeyeengine/pyeyeengine/engine_installation/install_script_generator.py
  cd ~/engine_package
  mkdir -p "packages"
  cp ~/engine_dev/pyeyeengine/packages/apt/*.deb ~/engine_package/packages/
  cp ~/engine_dev/pyeyeengine/packages/pip3/*.whl ~/engine_package/packages/
  cat ~/engine_dev/pyeyeengine/requirements.txt >> install.sh || { printf "\n\n*** Create Requirements Script Failed. Code: $? ***\n\n\n"; exit 4; }
  cp ~/engine_dev/pyeyeengine/utilities/Script/general_scripts/engine_handler.py .
  echo "cd .." >> install.sh
  echo "pip3 install pyeyeengine* >> /engineLog/engine_installation_dump.txt" >> install.sh
  echo 'echo "> Copying engine_handler.py"' >> install.sh
  echo 'rm /etc/engine_handler.py'
  echo 'cp engine_handler.py /etc/' >> install.sh
  echo 'echo "------- Engine installation Complete -------"' >> install.sh
  echo 'echo "Installation Finished" >> /engineLog/installation_finished.txt' >> install.sh
  echo "> Creating TAR archive"
  tar -czvf EnginePackage-$VERSION.$BUILD_NUMBER$CONFIGURATION.tgz * || { printf "\n\n*** Create TAR Archive Failed. Code: $? ***\n\n\n"; exit 7; }
  cp EnginePackage* ~/
  rm -rf ~/engine_dev
  rm -rf ~/engine_package
EOSSH

echo "> Getting EnginePackage from host"
mkdir -p $WORKSPACE/whlFile/
rm $WORKSPACE/whlFile/*
scp root@$WHL_FILE_COMPILER:~/EnginePackage*.tgz $WORKSPACE/whlFile/ || { printf "\n\n*** Copy EnginePackage Failed. Code: $? ***\n\n\n"; exit 3; }