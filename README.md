===============================================================================
--------------Script1----------------
File: par\_gui.py
Function: plot data in file from ucdf

```shell
cd ~/uos/install
bin/uos_replay-dump xx.txt  data/xxxxx.ucdf
python par_gui.py --log xx.txt
```
===============================================================================

===============================================================================
--------------Script2----------------
File: plot.sh
Function: plot data from ucdf directly
```shell
cd ~/uos/install
bash plot.sh -f ~/Downloads/log_2022-08-30-17-04-16/uos_20220830172612.ucdf
```
or
```shell
cd ~/uos/install
bash plot.sh -f ~/Downloads/log_2022-08-30-17-04-16/uos_20220830172612.ucdf
```

==============================================================================
