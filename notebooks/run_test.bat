@echo off
echo Activating torch-gpu conda environment and running model test...
call conda activate torch-gpu
python test_model.py
pause
