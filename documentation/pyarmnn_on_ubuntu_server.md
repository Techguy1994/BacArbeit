# Download and installing pyarmnn package on the raspberry pi (ubntu server version)

ToDO: 
keyboard interruption
path in argparse
keyboard change to german
and setup wifi without ethernet

At the moment Ubuntu 20.04.3 LTS version is the latest ubuntu server version with LTS support and pyarmnn has been tested on in this document. 

With this version installed on a microSD card and put into the raspberry pi 4 the first task is to connect to the internet via the Ethernet as the Network-Manager is not installed on ubuntu server.

Standard login data at first boot up is: 
```bash
login: ubuntu
password: ubuntu
```

With the Ethernet Connection (be aware the default language is english) 
```bash
  sudo apt update
  sudo apt upgrade
  sudo apt install network-manager
  sudo apt install net-tools
```

With the Network-Manger installed write
```bash
nmtui
```

A GUI opens up and choose create a connection and choose the desired Wifi-Connection and type the Wifi password. With this command check the ip-adress
```bash
ifconfig
```

With this you can access the Pi wireless with an SSH connection.

## Installing Pyarmnn for the Raspberry Pi 
This instruction is based on the Instruction from this github page: https://github.com/ARM-software/armnn/blob/branches/armnn_21_08/InstallationViaAptRepository.md
As for ease of use, here is a more compact form of it but is as of the content of it 100% the same

Adding PPA to the sources

```bash
sudo apt install software-properties-common
sudo add-apt-repository ppa:armnn/ppa
sudo apt update
```

With 
```bash
 apt-cache search libarmnn
 ```
 
 you can find the latest version as of the number with the packages you want to install. The latest one at the moment is Version 27
 
 ```bash
 export ARMNN_MAJOR_VERSION=27
 ```
 
 The rest is exactly as stated in the provided Github page.
 
  ```bash
sudo apt-get install -y python3-pyarmnn libarmnn-cpuacc-backend${ARMNN_MAJOR_VERSION} libarmnn-gpuacc-backend${ARMNN_MAJOR_VERSION} libarmnn-cpuref-backend${ARMNN_MAJOR_VERSION}
# Verify installation via python:
python3 -c "import pyarmnn as ann;print(ann.GetVersion())" 
 ```
 
If the last command return a version number, then pyarmnn has been successfully installed.
Depending on the script there might be python packages which are missing from the OS. These can be installed normally as with other Ubuntu versions.
