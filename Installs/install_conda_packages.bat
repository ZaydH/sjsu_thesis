echo Updating Python to the latest 2.x version
conda update python
echo Python update complete.

echo 
echo 
echo Installing Python packages.
conda install pip -y
conda install Pillow -y
conda install enum34 -y
conda install docutils -y
conda install sphinx -y
conda install numpy -y
conda install scitkit-image -y
conda install scipy -y
conda install scikit-learn -y


echo.
echo.
echo. End of Package Installation.
pause
goto:eof
