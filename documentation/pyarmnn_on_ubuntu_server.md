# Download and installing pyarmnn package on the raspberry pi (ubntu server version)

ToDO: 
keyboard interruption
path in argparse
keyboard change to german
and setup wifi without ethernet
tflite_runtime adding to the script

At the moment Ubuntu 20.04.3 LTS version is the latest ubuntu server version with LTS support and pyarmnn has been tested on in this document. 

With this version installed on a microSD card and put into the raspberry pi 4 the first task is to connect to the internet via the Ethernet as the Network-Manager is not installed on ubuntu server.

Standard login data at first boot up is: 
```bash
login: ubuntu
password: ubuntu
```

After first boot up some packages are installed in the background, this usually takes 20-30 minutes. This is the time to wait as no changes are possible.

A first optional task is to change the keyboard layout from english to german.

Source: https://www.manthanhd.com/2013/09/27/changing-keyboard-layout-in-ubuntu-server-linux-how-to/

```bash
sudo dpkg-reconfigure keyboard-configuration
```

![grafik](https://user-images.githubusercontent.com/31360730/151374375-cc478d24-8082-46b0-a49b-714310037b4b.png)

Press Enter on the keyboard to accept and get to the next page

![grafik](https://user-images.githubusercontent.com/31360730/151375061-47135d9d-0fac-4f9b-9107-3ec84a362d04.png)

then change the required language. The next pages are not required for the language settings.

## Connecting to Wifi

There are many possibilites how to connect to the Wifi. I will show 2 of them. 

## 1. Connecting to Wifi via the network-manager (Ethernet Connection required) 

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

## 2. Editing netplan configuration file
source: https://www.adminscave.com/how-to-connect-to-wifi-from-the-terminal-in-ubuntu-linux/

This method is based on the link above, but held shorter. If there any problems/questions you can read through the link.

We need the the name of the Connection from the OS

```bash
ubuntu@ubuntu:~$ ls /sys/class/net
eth0  lo  wlan0
```

It returns a list internect connection type. For Wifi the correct one is wlan0.

Now the proper yaml file has to be edited to include wlan0 and the SSID and password of the Wifi Connection

```bash
sudo nano /etc/netplan/50-cloud-init.yaml
```

Usually it should only contain the Ethernet enty. Now the Wifi entry has to be added. In the end it has to look like this

```bash
# This file is generated from information provided by the datasource.  Changes
# to it will not persist across an instance reboot.  To disable cloud-init's
# network configuration capabilities, write a file
# /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg with the following:
# network: {config: disabled}
network:
    ethernets:
        eth0:
            dhcp4: true
            optional: true
    version: 2
    wifis:
        wlan0:
            dhcp4: true
            optional: true
            access-points:
                "SSID_name":
                    password: "Wifi password"
```

In the quotes replace SSID_name and Wifi password with the actual connection name and password. Be aware that the idention has to be exact and do not use tabs, it will not work.

To save the changes hit Ctrl+X and accept the changes with a y. Then exit the file.

With these commands you apply the changes.

```bash
sudo netplan generate
sudo netplan apply
``` 

After a few minutes check with 

```bash
ifconfig wlan0
```
if there is an ip-adress shown. This would be the number after inet.
When inet is not shown type this command and repeat the steps for netplan. 

```bash
sudo systemctl start wpa_supplicant
sudo netplan generate
sudo netplan apply
```

If it stills does not work after a few minutes then shutdown the the Raspberry Pi with

```bash
shutdown now
```

and turn it on again. Check if there is an internet connection and when not run these commands again and check again

```bash
sudo netplan generate
sudo netplan apply
``` 

If nothing worked, check the file if everything is written correctly.
When you have an internet connection write or memorize the ip adress for the SSH connection

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
